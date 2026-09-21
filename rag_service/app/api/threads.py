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
- GET /api/threads/{thread_id}/embeddings-projection - Get 3D embedding projection
"""

import asyncio
import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.agent.prompting import (
    get_tool_catalog,
    normalize_tool_instructions,
)
from app.product_orchestration.repository import AgentWorkflowRepository
from app.product_orchestration.builtin_workflows import builtin_workflow_keys
from app.product_orchestration.workflow_runtime import (
    default_agent_workflow_key,
    workflow_is_chat_eligible,
    workflow_supports_replans,
)
from app.time_utils import iso_utc_z
from app.runtime.builder_registry import builder_for_definition
from app.runtime.catalog import definition_from_workflow
from app.db import (
    EmbeddingReadinessStatus,
    FileSourceType,
    MemoryScopeType,
    ProcessStatus,
    delete_thread,
    assign_thread_to_project,
    ensure_default_project,
    get_file_status,
    get_file,
    get_project,
    get_thread,
    get_thread_files,
    get_effective_thread_files,
    get_thread_settings,
    get_scoped_indexing_status,
    is_file_accessible_to_thread,
    list_threads,
    update_thread,
    update_thread_settings,
)
from app.db.vector import get_vector_db
from app.models.llm_server_client import check_embedding_model_ready, merge_thread_settings
from app.db.vector.config import VectorDBQueryError
from app.models.requests import (
    EmbeddingProjectionEdge,
    EmbeddingProjectionPoint,
    EmbeddingProjectionResponse,
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
from app.services.embedding_model_service import (
    EmbeddingModelResolutionError,
    EmbeddingModelUnavailableError,
    require_thread_embedding_ready,
)
from app.services.embedding_projection_service import (
    compute_projection_edges,
    merge_embedding_family_points,
    project_embeddings_3d,
    resolve_embedding_source_families,
)
from app.services.memory_policy import LOCAL_USER_MEMORY_SCOPE_ID
from app.rag.indexer import trigger_reembed_for_missing_sources
from app.services.file_cleanup_service import cleanup_detached_file
from app.services.memory_service import hard_delete_thread_memory_resources
from app.services.thread_management_service import (
    ForkMessageNotFoundError,
    TargetProjectEmbeddingModelMismatchError,
    SourceThreadNotFoundError,
    fork_thread,
    repair_thread_documents_meta,
)

router = APIRouter(tags=["threads"])

EMBEDDING_PROJECTION_DEFAULT_LIMIT = 300
EMBEDDING_PROJECTION_MAX_LIMIT = 1000


async def _settings_workflow_supports_replans(settings: dict) -> bool:
    agent_workflow = settings.get("agent_workflow")
    workflow_id = agent_workflow.get("workflow_id") if isinstance(agent_workflow, dict) else None
    if not isinstance(workflow_id, str) or not workflow_id:
        return False
    workflow = await AgentWorkflowRepository().get_workflow(workflow_id, include_custom=True)
    spec = workflow.spec_json if workflow and isinstance(workflow.spec_json, dict) else {}
    return workflow_supports_replans(spec)


async def _resolve_chat_workflow(workflow_id: str):
    repo = AgentWorkflowRepository()
    await repo.seed_builtin_workflows()
    workflow = await repo.get_workflow(
        workflow_id,
        include_custom=workflow_id not in builtin_workflow_keys(),
    )
    return workflow if workflow and workflow_is_chat_eligible(workflow.spec_json) else None


async def _normalize_chat_workflow_setting(settings: dict) -> tuple[dict, Optional[dict]]:
    agent_workflow = settings.get("agent_workflow")
    workflow_id = agent_workflow.get("workflow_id") if isinstance(agent_workflow, dict) else None
    workflow_id = str(workflow_id or default_agent_workflow_key())
    if await _resolve_chat_workflow(workflow_id):
        return settings, None
    return settings, {
        "valid": False,
        "code": "workflow_not_available_for_chat",
        "requested_workflow_id": workflow_id,
    }


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


async def _thread_file_payloads(thread, files) -> list[dict]:
    payloads = []
    for file in files:
        status = await get_file_status(file.file_hash) or {}
        scoped_indexing = get_scoped_indexing_status(
            status,
            thread.embedding_model,
            thread.id if file.association_scope == "thread" else None,
        )
        sections = (status.get("parsing") or {}, scoped_indexing)
        failed = next(
            (section for section in sections if ProcessStatus.is_failed(section.get("status"))),
            None,
        )
        if failed:
            processing_status = ProcessStatus.FAILED.value
            processing_error = str(failed.get("error") or "Processing failed")
        elif all(ProcessStatus.is_completed(section.get("status")) for section in sections):
            processing_status = ProcessStatus.COMPLETED.value
            processing_error = None
        else:
            processing_status = ProcessStatus.PENDING.value
            processing_error = None
        payloads.append({
            "file_hash": file.file_hash,
            "file_name": file.file_name,
            "file_path": file.file_path,
            "source_type": file.source_type,
            "association_scope": file.association_scope,
            "is_project_knowledge": file.is_project_knowledge,
            "added_at": iso_utc_z(file.added_at),
            "processing_status": processing_status,
            "processing_error": processing_error,
        })
    return payloads


async def _delete_thread_resources(thread_id: str) -> bool:
    """
    Delete a thread, its thread-scoped vectors, and any files detached by that delete.

    Returns False when the thread does not exist.
    """
    thread = await get_thread(thread_id)
    if not thread:
        return False

    files = await get_thread_files(thread_id)
    from app.runtime.cleanup import cleanup_runs
    from app.services.agent_task_repository import list_task_runtime_runs_for_threads
    from app.services.task_artifact_service import delete_task_resources_for_threads
    outcomes = await cleanup_runs(
        await list_task_runtime_runs_for_threads([thread_id])
    )
    if any(not outcome.owner_deletion_allowed for outcome in outcomes):
        raise RuntimeError("Runtime continuation cleanup was not confirmed")
    await delete_task_resources_for_threads([thread_id])

    db = get_vector_db()
    await db.delete_thread_data(thread_id)
    await hard_delete_thread_memory_resources(thread_id)
    from app.services.embedding_materialization_service import cancel_embedding_jobs_for_scope
    await cancel_embedding_jobs_for_scope(thread_id)

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
        payload = PromptDefaults(**defaults).model_dump(mode="json")
        return {
            "tools": [ToolCatalogEntry(**t).model_dump(mode="json") for t in get_tool_catalog()],
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
        workflow = await repo.get_workflow(
            workflow_id,
            include_custom=workflow_id not in supported_builtin_workflow_keys,
        )
        if workflow is None:
            raise HTTPException(status_code=404, detail={"code": "agent_workflow_not_found"})
        if not workflow_is_chat_eligible(workflow.spec_json):
            raise HTTPException(status_code=422, detail={"code": "agent_workflow_not_chat_eligible"})
        spec = workflow.spec_json if workflow and isinstance(workflow.spec_json, dict) else {}
        definition = definition_from_workflow(workflow)
        prompt = await builder_for_definition(definition).prompt_preview(
            definition,
            spec,
            {
                "context_window": req.context_window,
                "system_role": req.system_role or "",
                "tool_instructions": tool_instructions,
                "custom_instructions": req.custom_instructions or "",
                "use_web_search": req.use_web_search,
                "client_timezone": req.client_timezone,
                "client_locale": req.client_locale,
                "client_now_iso": req.client_now_iso,
            },
        )
        return {"prompt": prompt}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threads")
async def create_thread_endpoint(req: ThreadCreateRequest):
    """Create a new chat thread."""
    try:
        # Create thread in the selected project and inherit its immutable model.
        from app.db import create_thread
        project_id = req.project_id
        if project_id:
            project = await get_project(project_id)
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
        else:
            project = await ensure_default_project()
            project_id = project.id
        thread = await create_thread(req.name, project_id)

        return _thread_payload(thread)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
    except TargetProjectEmbeddingModelMismatchError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads/{thread_id}/embeddings-projection", response_model=EmbeddingProjectionResponse)
async def get_thread_embeddings_projection_endpoint(
    thread_id: str,
    file_hash: Optional[str] = None,
    source_kind: Optional[str] = None,
    source_family: Optional[str] = None,
    limit: int = EMBEDDING_PROJECTION_DEFAULT_LIMIT,
):
    """Return 3D-projected embeddings for the thread workspace viewer."""
    if limit <= 0 or limit > EMBEDDING_PROJECTION_MAX_LIMIT:
        raise HTTPException(
            status_code=400,
            detail=f"limit must be between 1 and {EMBEDDING_PROJECTION_MAX_LIMIT}",
        )
    try:
        families = resolve_embedding_source_families(source_family)
        context = await require_thread_embedding_ready(thread_id)
        files = await get_effective_thread_files(thread_id)
        file_name_lookup = {
            file.file_hash: getattr(file, "file_name", None) or file.file_hash
            for file in files
        }
        accessible_hashes = list(file_name_lookup.keys())
        target_hashes = accessible_hashes
        if file_hash and "documents" in families:
            if file_hash not in file_name_lookup:
                raise HTTPException(status_code=404, detail="File is not accessible to this thread")
            target_hashes = [file_hash]

        db = get_vector_db()
        family_points: dict[str, list[dict]] = {}
        if "documents" in families and target_hashes:
            family_points["documents"] = await db.get_thread_vector_points(
                context.embedding_model,
                file_hashes=target_hashes,
                source_kind=source_kind,
                limit=limit,
            )
        if "chat" in families:
            family_points["chat"] = await db.get_thread_chat_vector_points(
                context.embedding_model,
                thread_id=thread_id,
                limit=limit,
            )
        if "web_search" in families:
            family_points["web_search"] = await db.get_thread_web_search_vector_points(
                context.embedding_model,
                thread_id=thread_id,
                limit=limit,
            )
        if "memory" in families:
            project_id = getattr(context.project, "id", None)
            scope_filters = [
                {"scope_type": MemoryScopeType.THREAD.value, "scope_id": thread_id},
            ]
            if project_id:
                scope_filters.append(
                    {"scope_type": MemoryScopeType.PROJECT.value, "scope_id": str(project_id)}
                )
            scope_filters.append(
                {
                    "scope_type": MemoryScopeType.USER.value,
                    "scope_id": LOCAL_USER_MEMORY_SCOPE_ID,
                }
            )
            family_points["memory"] = await db.get_thread_memory_vector_points(
                context.embedding_model,
                scope_filters=scope_filters,
                limit=limit,
            )

        raw_points, truncated = merge_embedding_family_points(
            family_points,
            family_order=families,
            limit=limit,
        )
        vectors = [point["vector"] for point in raw_points]
        coordinates = project_embeddings_3d(vectors)
        response_points = [
            EmbeddingProjectionPoint(
                id=str(point["id"]),
                x=coords[0],
                y=coords[1],
                z=coords[2],
                chunk_id=point.get("chunk_id"),
                file_hash=str(point.get("file_hash") or ""),
                file_name=(
                    file_name_lookup.get(str(point.get("file_hash") or ""))
                    or point.get("file_name")
                    or point.get("title")
                ),
                text=str(point.get("text") or ""),
                page_start=point.get("page_start"),
                page_end=point.get("page_end"),
                pages=point.get("pages"),
                source_kind=point.get("source_kind"),
                table_id=point.get("table_id"),
                section_id=point.get("section_id"),
            )
            for point, coords in zip(raw_points, coordinates)
        ]
        raw_edges = compute_projection_edges(raw_points, vectors)
        response_edges = [
            EmbeddingProjectionEdge(
                id=edge["id"],
                source=edge["source"],
                target=edge["target"],
                label=edge.get("label"),
                kind=edge["kind"],
                score=edge.get("score"),
            )
            for edge in raw_edges
        ]
        return EmbeddingProjectionResponse(
            thread_id=thread_id,
            embedding_model=context.embedding_model,
            point_count=len(response_points),
            truncated=truncated,
            points=response_points,
            edges=response_edges,
        )
    except HTTPException:
        raise
    except EmbeddingModelResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "embedding_model_unavailable", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VectorDBQueryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/threads/{thread_id}")
async def get_thread_endpoint(thread_id: str):
    """Get a specific thread by ID."""
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        files = await get_effective_thread_files(thread_id)
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
            "files": await _thread_file_payloads(thread, files),
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
        try:
            thread = await assign_thread_to_project(thread_id, req.project_id)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
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
        settings, workflow_validation = await _normalize_chat_workflow_setting(settings)
        settings["tool_instructions"] = normalize_tool_instructions(
            settings.get("tool_instructions", {})
        )
        return ThreadSettingsResponse(**settings, agent_workflow_validation=workflow_validation)
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
        updates = req.model_dump(exclude_none=True)
        next_settings = {**current, **updates}
        requested_agent_workflow = next_settings.get("agent_workflow")
        requested_workflow_id = (
            requested_agent_workflow.get("workflow_id")
            if isinstance(requested_agent_workflow, dict)
            else None
        )
        if not isinstance(requested_workflow_id, str) or not requested_workflow_id.strip():
            raise HTTPException(status_code=400, detail="A valid chat agent workflow is required")
        if not await _resolve_chat_workflow(requested_workflow_id):
            raise HTTPException(status_code=400, detail="Agent workflow is not available for chat")
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

        from app.services.embedding_tokenizer import EmbeddingTokenizerUnavailableError, resolve_embedding_tokenizer
        try:
            resolve_embedding_tokenizer(thread.embedding_model)
        except EmbeddingTokenizerUnavailableError as exc:
            return {
                "thread_id": thread_id,
                "status": EmbeddingReadinessStatus.BLOCKED.value,
                "stats": _empty_thread_stats(),
                "embedding_model_ready": True,
                "error": {"code": "embedding_tokenizer_unavailable", "message": str(exc)},
            }

        db = get_vector_db()
        from app.services.document_projection_service import evaluate_retrieval_readiness
        from app.services.embedding_materialization_service import enqueue_document_embedding_if_needed

        # Track files list for stats query
        files = []

        if file_hash:
            if not await is_file_accessible_to_thread(thread_id, file_hash):
                raise HTTPException(status_code=404, detail="File not found")
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
            file_record = await get_file(file_hash)
            scoped_indexing = get_scoped_indexing_status(
                file_status,
                embedding_model=thread.embedding_model,
                thread_id=thread_id,
            )
            indexing_status = scoped_indexing.get("status", ProcessStatus.UNKNOWN.value)
            if file_record is not None and FileSourceType.uses_pdf_conversion(getattr(file_record, "source_type", None)):
                # Conversion and retrieval readiness are authoritative.  Run
                # this gate even while file_status is pending so conversion
                # completion can create the exact embedding target without
                # requiring a second access or a stale status transition.
                readiness = await evaluate_retrieval_readiness(file_hash, thread.embedding_model, thread_id=thread_id)
                await enqueue_document_embedding_if_needed(
                    file_hash=file_hash,
                    thread_id=thread_id,
                    embedding_model=thread.embedding_model,
                    file_name=getattr(file_record, "file_name", None),
                    readiness=readiness,
                )
                status = EmbeddingReadinessStatus.READY.value if readiness.get("ready") else EmbeddingReadinessStatus.NOT_READY.value
            else:
                status = EmbeddingReadinessStatus.READY.value if ProcessStatus.is_completed(indexing_status) else EmbeddingReadinessStatus.NOT_READY.value
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
                    file_ready = ProcessStatus.is_completed(indexing_status)
                    if FileSourceType.uses_pdf_conversion(getattr(f, "source_type", None)):
                        readiness = await evaluate_retrieval_readiness(f.file_hash, thread.embedding_model, thread_id=thread_id)
                        await enqueue_document_embedding_if_needed(
                            file_hash=f.file_hash,
                            thread_id=thread_id,
                            embedding_model=thread.embedding_model,
                            file_name=getattr(f, "file_name", None),
                            readiness=readiness,
                        )
                        file_ready = bool(readiness.get("ready"))
                    if not file_ready:
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
