"""
Threads API Module - Thread management endpoints.

Endpoints:
- GET /api/threads/prompt-tools - Get prompt tools and defaults
- POST /api/threads/prompt-preview - Get prompt preview
- POST /api/threads - Create thread
- GET /api/threads - List threads
- POST /api/threads/bulk/delete - Delete multiple threads
- POST /api/threads/{thread_id}/fork - Fork thread
- GET /api/threads/{thread_id} - Get thread
- PUT /api/threads/{thread_id} - Update thread
- GET /api/threads/{thread_id}/settings - Get thread settings
- PUT /api/threads/{thread_id}/settings - Update thread settings
- DELETE /api/threads/{thread_id} - Delete thread
- GET /api/threads/{thread_id}/indexing/status - Get thread indexing status
"""

import asyncio
import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.agent.prompting import (
    get_tool_catalog,
    normalize_tool_instructions,
)
from app.agent_workflows.prompting import build_agent_workflow_prompt_preview
from app.agent_workflows.repository import AgentWorkflowRepository
from app.agent_workflows.builtin_workflows import builtin_workflow_keys
from app.agent_workflows.workflow_runtime import default_agent_workflow_key, workflow_supports_replans
from app.time_utils import iso_utc_z
from app.db import (
    EmbeddingReadinessStatus,
    ProcessStatus,
    delete_thread,
    assign_thread_to_project,
    ensure_default_project,
    get_file_status,
    get_project,
    get_thread,
    get_thread_files,
    get_thread_settings,
    get_scoped_indexing_status,
    list_threads,
    update_thread,
    update_thread_settings,
)
from app.db.vector import get_vector_db
from app.models.llm_server_client import (
    LOCAL_EMBEDDING_MODEL,
    check_embedding_model_ready,
    merge_thread_settings,
)
from app.models.requests import (
    PromptDefaults,
    PromptPreviewRequest,
    ThreadBulkDeleteRequest,
    ThreadBulkDeleteResponse,
    ThreadCreateRequest,
    ThreadForkRequest,
    ThreadProjectUpdateRequest,
    ThreadSettingsResponse,
    ThreadSettingsUpdateRequest,
    ThreadUpdateRequest,
    ToolCatalogEntry,
)
from app.rag.indexer import trigger_reembed_for_missing_sources
from app.services.file_cleanup_service import cleanup_detached_file
from app.services.thread_management_service import (
    ForkMessageNotFoundError,
    SourceThreadNotFoundError,
    fork_thread,
    repair_thread_documents_meta,
)

router = APIRouter(tags=["threads"])


async def _settings_workflow_supports_replans(settings: dict) -> bool:
    agent_workflow = settings.get("agent_workflow")
    workflow_id = agent_workflow.get("workflow_id") if isinstance(agent_workflow, dict) else None
    if not isinstance(workflow_id, str) or not workflow_id:
        return False
    workflow = await AgentWorkflowRepository().get_workflow(workflow_id, include_custom=True)
    spec = workflow.spec_json if workflow and isinstance(workflow.spec_json, dict) else {}
    return workflow_supports_replans(spec)


def _empty_thread_stats() -> dict:
    return {
        "total_documents": 0,
        "total_chunks": 0,
        "total_chars": 0,
        "documents": {},
    }


def _public_thread_settings(settings: Optional[dict]) -> dict:
    """Return persisted thread settings without stale/unknown settings keys."""
    if not isinstance(settings, dict) or not settings:
        return {}
    allowed_keys = set(merge_thread_settings({}).keys())
    return {key: value for key, value in settings.items() if key in allowed_keys}


def _thread_payload(thread) -> dict:
    return {
        "id": thread.id,
        "project_id": getattr(thread, "project_id", None),
        "name": thread.name,
        "embedding_model": thread.embedding_model,
        "settings": _public_thread_settings(thread.settings),
        "thread_metadata": thread.thread_metadata if thread.thread_metadata else {},
        "created_at": iso_utc_z(thread.created_at),
    }


async def _delete_thread_resources(thread_id: str) -> bool:
    """
    Delete a thread, its thread-scoped vectors, and any files detached by that delete.

    Returns False when the thread does not exist.
    """
    thread = await get_thread(thread_id)
    if not thread:
        return False

    files = await get_thread_files(thread_id)

    db = get_vector_db()
    await db.delete_thread_data(thread_id)

    deleted = await delete_thread(thread_id)
    if not deleted:
        return False

    for file in files:
        await cleanup_detached_file(file.file_hash, thread_id, thread.embedding_model)

    return True


@router.get("/threads/prompt-tools")
async def prompt_tools_endpoint():
    """Return user-facing tool aliases and default prompts for prompt customization UI."""
    try:
        defaults = merge_thread_settings({})
        defaults["tool_instructions"] = normalize_tool_instructions(
            defaults.get("tool_instructions", {})
        )
        payload = PromptDefaults(**defaults).dict()
        return {
            "tools": [ToolCatalogEntry(**t).dict() for t in get_tool_catalog()],
            "defaults": payload,
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threads/prompt-preview")
async def prompt_preview_endpoint(req: PromptPreviewRequest):
    """Return the fully composed system prompt preview from the backend source of truth."""
    try:
        tool_instructions = normalize_tool_instructions(req.tool_instructions or {})
        requested_workflow = req.agent_workflow_id
        if not requested_workflow and isinstance(req.agent_workflow, dict):
            requested_workflow = req.agent_workflow.get("workflow_id")
        repo = AgentWorkflowRepository()
        await repo.seed_builtin_workflows()
        supported_builtin_workflow_keys = builtin_workflow_keys()
        workflow_id = requested_workflow if requested_workflow else default_agent_workflow_key()
        workflow = await repo.get_workflow(workflow_id, include_custom=workflow_id not in supported_builtin_workflow_keys)
        if workflow is None:
            workflow = await repo.get_workflow(default_agent_workflow_key())
        spec = workflow.spec_json if workflow and isinstance(workflow.spec_json, dict) else {}
        runtime = spec.get("runtime") if isinstance(spec.get("runtime"), dict) else {}
        prompt = build_agent_workflow_prompt_preview(
            prompt_profile=str(runtime.get("prompt_preview") or "router"),
            context_window=req.context_window,
            system_role=req.system_role or "",
            tool_instructions=tool_instructions,
            custom_instructions=req.custom_instructions or "",
            use_web_search=req.use_web_search,
            client_timezone=req.client_timezone,
            client_locale=req.client_locale,
            client_now_iso=req.client_now_iso,
        )
        return {"prompt": prompt}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threads")
async def create_thread_endpoint(req: ThreadCreateRequest):
    """Create a new chat thread."""
    try:
        embedding_model = (req.embedding_model or "").strip() or LOCAL_EMBEDDING_MODEL
        if not embedding_model:
            raise HTTPException(
                status_code=400,
                detail="embedding_model is required (set LOCAL_EMBEDDING_MODEL or pass embedding_model).",
            )
        # Create thread in database
        from app.db import create_thread
        project_id = req.project_id
        if project_id:
            project = await get_project(project_id)
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
        else:
            project = await ensure_default_project()
            project_id = project.id
        thread = await create_thread(req.name, embedding_model, project_id=project_id)

        return _thread_payload(thread)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads")
async def list_threads_endpoint():
    """List all threads with message and file counts."""
    try:
        threads = await list_threads()
        for thread in threads:
            thread["settings"] = _public_thread_settings(thread.get("settings"))
        return {"threads": threads}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threads/bulk/delete", response_model=ThreadBulkDeleteResponse)
async def bulk_delete_threads_endpoint(req: ThreadBulkDeleteRequest):
    """Delete multiple threads, returning per-thread results."""
    seen_thread_ids = set()
    thread_ids = []
    for thread_id in req.thread_ids:
        normalized = (thread_id or "").strip()
        if not normalized or normalized in seen_thread_ids:
            continue
        seen_thread_ids.add(normalized)
        thread_ids.append(normalized)

    if not thread_ids:
        raise HTTPException(
            status_code=400,
            detail="thread_ids must contain at least one thread ID",
        )
    if len(thread_ids) > 100:
        raise HTTPException(
            status_code=400,
            detail="thread_ids cannot contain more than 100 unique thread IDs",
        )

    deleted_thread_ids = []
    not_found_thread_ids = []
    failed_thread_ids = []

    for thread_id in thread_ids:
        try:
            deleted = await _delete_thread_resources(thread_id)
            if deleted:
                deleted_thread_ids.append(thread_id)
            else:
                not_found_thread_ids.append(thread_id)
        except Exception as e:
            traceback.print_exc()
            failed_thread_ids.append({"thread_id": thread_id, "error": str(e)})

    return ThreadBulkDeleteResponse(
        deleted_thread_ids=deleted_thread_ids,
        not_found_thread_ids=not_found_thread_ids,
        failed_thread_ids=failed_thread_ids,
    )


@router.post("/threads/{thread_id}/fork")
async def fork_thread_endpoint(thread_id: str, req: ThreadForkRequest):
    """Fork a thread from an optional message point."""
    try:
        result = await fork_thread(
            source_thread_id=thread_id,
            message_id=req.message_id,
            name=req.name,
            target_project_id=req.target_project_id,
            memory_copy_mode=req.memory_copy_mode,
        )
        thread = result["thread"]
        files = result["files"]
        asyncio.create_task(
            trigger_reembed_for_missing_sources(
                thread_id=thread.id,
                embedding_model=thread.embedding_model,
                file_hashes=[f.file_hash for f in files],
            )
        )
        return _thread_payload(thread)
    except SourceThreadNotFoundError:
        raise HTTPException(status_code=404, detail="Thread not found")
    except ForkMessageNotFoundError:
        raise HTTPException(
            status_code=400,
            detail="Fork message not found in source thread",
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads/{thread_id}")
async def get_thread_endpoint(thread_id: str):
    """Get a specific thread by ID."""
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        files = await get_thread_files(thread_id)
        embedding_model_ready = await check_embedding_model_ready(thread.embedding_model)
        stats = _empty_thread_stats()
        stats_unavailable_reason = None

        if embedding_model_ready:
            await repair_thread_documents_meta(thread_id, thread.embedding_model, files)
            asyncio.create_task(
                trigger_reembed_for_missing_sources(
                    thread_id=thread_id,
                    embedding_model=thread.embedding_model,
                )
            )
            # Proactively ensure all collections exist for this thread's embedding model
            asyncio.create_task(
                get_vector_db().collection_manager.ensure_collections_for_thread(
                    embedding_model=thread.embedding_model
                )
            )
            db = get_vector_db()
            stats = await db.get_thread_stats(
                thread_id=thread_id,
                file_hashes=[f.file_hash for f in files],
                embedding_model=thread.embedding_model,
            )
        else:
            stats_unavailable_reason = "Embedding model is not ready"

        return {
            "id": thread.id,
            "project_id": getattr(thread, "project_id", None),
            "name": thread.name,
            "embedding_model": thread.embedding_model,
            "settings": _public_thread_settings(thread.settings),
            "thread_metadata": getattr(thread, "thread_metadata", None) or {},
            "created_at": iso_utc_z(thread.created_at),
            "files": [
                {
                    "file_hash": f.file_hash,
                    "file_name": f.file_name,
                    "file_path": f.file_path,
                    "source_type": f.source_type,
                }
                for f in files
            ],
            "stats": stats,
            "embedding_model_ready": embedding_model_ready,
            "stats_unavailable_reason": stats_unavailable_reason,
            "file_count": len(files),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/threads/{thread_id}")
async def update_thread_endpoint(thread_id: str, req: ThreadUpdateRequest):
    """
    Update a thread's name.
    Note: embedding_model cannot be changed once set.
    """
    try:
        thread = await update_thread(thread_id, req.name)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        return _thread_payload(thread)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/threads/{thread_id}/project")
async def update_thread_project_endpoint(thread_id: str, req: ThreadProjectUpdateRequest):
    """Move a thread into a project."""
    try:
        thread = await assign_thread_to_project(thread_id, req.project_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread or project not found")
        return _thread_payload(thread)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads/{thread_id}/settings")
async def get_thread_settings_endpoint(thread_id: str):
    """Get persisted chat settings for a thread."""
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        settings = merge_thread_settings(await get_thread_settings(thread_id))
        settings["tool_instructions"] = normalize_tool_instructions(
            settings.get("tool_instructions", {})
        )
        return ThreadSettingsResponse(**settings)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/threads/{thread_id}/settings")
async def update_thread_settings_endpoint(
    thread_id: str, req: ThreadSettingsUpdateRequest
):
    """Update persisted chat settings for a thread."""
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        current = merge_thread_settings(await get_thread_settings(thread_id))
        updates = req.dict(exclude_none=True)
        next_settings = {**current, **updates}
        if not await _settings_workflow_supports_replans(next_settings):
            next_settings.pop("replans", None)
        next_settings["tool_instructions"] = normalize_tool_instructions(
            next_settings.get("tool_instructions", {})
        )
        persisted = await update_thread_settings(thread_id, next_settings)
        if persisted is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        merged = merge_thread_settings(persisted)
        merged["tool_instructions"] = normalize_tool_instructions(
            merged.get("tool_instructions", {})
        )
        return ThreadSettingsResponse(**merged)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/threads/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    """
    Delete a thread, its messages, file associations, and vector data.
    """
    try:
        deleted = await _delete_thread_resources(thread_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Thread not found")

        return {"status": "deleted", "thread_id": thread_id}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads/{thread_id}/indexing/status")
async def get_thread_index_status_endpoint(thread_id: str, file_hash: Optional[str] = None):
    """
    Check indexing status for a thread (or specific file in thread).
    Uses file_status for per-file indexing state.
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        embedding_model_ready = await check_embedding_model_ready(thread.embedding_model)
        if not embedding_model_ready:
            return {
                "thread_id": thread_id,
                "status": EmbeddingReadinessStatus.BLOCKED.value,
                "stats": _empty_thread_stats(),
                "embedding_model_ready": False,
            }

        db = get_vector_db()

        # Track files list for stats query
        files = []

        if file_hash:
            # Check specific file using file_status
            file_status = await get_file_status(file_hash)
            # Handle case where file doesn't exist yet (returns empty dict)
            if not file_status:
                return {
                    "thread_id": thread_id,
                    "status": EmbeddingReadinessStatus.NOT_READY.value,
                    "stats": _empty_thread_stats(),
                    "embedding_model_ready": embedding_model_ready,
                }
            scoped_indexing = get_scoped_indexing_status(
                file_status,
                embedding_model=thread.embedding_model,
                thread_id=thread_id,
            )
            indexing_status = scoped_indexing.get("status", ProcessStatus.UNKNOWN.value)
            if ProcessStatus.is_completed(indexing_status):
                status = EmbeddingReadinessStatus.READY.value
            elif ProcessStatus.is_failed(indexing_status):
                status = EmbeddingReadinessStatus.NOT_READY.value
            elif ProcessStatus.is_running(indexing_status):
                status = EmbeddingReadinessStatus.NOT_READY.value
            else:
                # Fallback to vector DB check for backward compatibility
                is_indexed = await db.has_file_indexed(thread_id, file_hash, thread.embedding_model)
                status = EmbeddingReadinessStatus.READY.value if is_indexed else EmbeddingReadinessStatus.NOT_READY.value
        else:
            # Check all files in thread using file_status
            files = await get_thread_files(thread_id)
            if not files:
                status = EmbeddingReadinessStatus.READY.value
            else:
                all_indexed = True
                for f in files:
                    file_status = await get_file_status(f.file_hash)
                    scoped_indexing = get_scoped_indexing_status(
                        file_status,
                        embedding_model=thread.embedding_model,
                        thread_id=thread_id,
                    )
                    indexing_status = scoped_indexing.get("status", ProcessStatus.UNKNOWN.value)
                    if not ProcessStatus.is_completed(indexing_status):
                        # Fallback to vector DB check for backward compatibility
                        if not await db.has_file_indexed(thread_id, f.file_hash, thread.embedding_model):
                            all_indexed = False
                            break
                status = EmbeddingReadinessStatus.READY.value if all_indexed else EmbeddingReadinessStatus.NOT_READY.value

        # Build file hashes list for stats query
        file_hashes = [file_hash] if file_hash else ([f.file_hash for f in files] if files else [])
        stats = await db.get_thread_stats(
            thread_id=thread_id,
            file_hashes=file_hashes,
            embedding_model=thread.embedding_model,
        )

        return {
            "thread_id": thread_id,
            "status": status,
            "stats": stats,
            "embedding_model_ready": embedding_model_ready,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
