"""
Threads API Module - Thread management endpoints.

Endpoints:
- GET /api/threads/prompt-tools - Get prompt tools and defaults
- POST /api/threads/prompt-preview - Get prompt preview
- POST /api/threads - Create thread
- GET /api/threads - List threads
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

from app.agent.agent import (
    build_system_prompt,
    get_tool_catalog,
    normalize_tool_instructions,
)
from app.db import (
    ProcessStatus,
    delete_thread,
    get_file_status,
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
    DEFAULT_EMBEDDING_MODEL,
    check_embed_model_ready,
    merge_thread_settings,
)
from app.models.requests import (
    PromptDefaults,
    PromptPreviewRequest,
    ThreadCreateRequest,
    ThreadSettingsResponse,
    ThreadSettingsUpdateRequest,
    ThreadUpdateRequest,
    ToolCatalogEntry,
)
from app.rag.indexer import trigger_reembed_for_missing_sources
from app.services.file_cleanup_service import cleanup_detached_file
from app.services.thread_management_service import repair_thread_documents_meta

router = APIRouter(tags=["threads"])


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
        prompt = build_system_prompt(
            context_window=req.context_window,
            system_role=req.system_role or "",
            tool_instructions=tool_instructions,
            custom_instructions=req.custom_instructions or "",
            use_web_search=req.use_web_search,
            intent_agent_ran=req.intent_agent_ran,
            reasoning_mode=req.reasoning_mode,
        )
        return {"prompt": prompt}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threads")
async def create_thread_endpoint(req: ThreadCreateRequest):
    """Create a new chat thread."""
    try:
        embed_model = (req.embed_model or "").strip() or DEFAULT_EMBEDDING_MODEL
        if not embed_model:
            raise HTTPException(
                status_code=400,
                detail="embed_model is required (set DEFAULT_EMBEDDING_MODEL or pass embed_model).",
            )
        # Create thread in database
        from app.db import create_thread
        thread = await create_thread(req.name, embed_model)

        return {
            "id": thread.id,
            "name": thread.name,
            "embed_model": thread.embed_model,
            "settings": thread.settings,
            "created_at": thread.created_at.isoformat(),
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads")
async def list_threads_endpoint():
    """List all threads with message and file counts."""
    try:
        threads = await list_threads()
        return {"threads": threads}
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
        await repair_thread_documents_meta(thread_id, thread.embed_model, files)
        asyncio.create_task(
            trigger_reembed_for_missing_sources(
                thread_id=thread_id,
                embedding_model_name=thread.embed_model,
            )
        )
        db = get_vector_db()
        stats = await db.get_thread_stats(
            thread_id=thread_id,
            file_hashes=[f.file_hash for f in files],
            embedding_model_name=thread.embed_model,
        )
        return {
            "id": thread.id,
            "name": thread.name,
            "embed_model": thread.embed_model,
            "settings": thread.settings,
            "created_at": thread.created_at.isoformat(),
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
    Note: embed_model cannot be changed once set.
    """
    try:
        thread = await update_thread(thread_id, req.name)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        return {
            "id": thread.id,
            "name": thread.name,
            "embed_model": thread.embed_model,
            "settings": thread.settings,
            "created_at": thread.created_at.isoformat(),
        }
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
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        files = await get_thread_files(thread_id)

        # Delete thread-scoped vector data first
        db = get_vector_db()
        await db.delete_thread_data(thread_id)

        # Delete from database
        from app.db import delete_thread as db_delete_thread
        deleted = await db_delete_thread(thread_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Thread not found")

        for file in files:
            await cleanup_detached_file(file.file_hash, thread_id, thread.embed_model)

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
    Now uses file_status column instead of thread_stats.
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        db = get_vector_db()
        embed_model_ready = await check_embed_model_ready(thread.embed_model)

        if file_hash:
            # Check specific file using file_status
            file_status = await get_file_status(file_hash)
            scoped_indexing = get_scoped_indexing_status(
                file_status,
                embedding_model=thread.embed_model,
                thread_id=thread_id,
            )
            indexing_status = scoped_indexing.get("status", ProcessStatus.UNKNOWN.value)
            if ProcessStatus.is_completed(indexing_status):
                status = "ready"
            elif ProcessStatus.is_failed(indexing_status):
                status = "not_ready"
            elif ProcessStatus.is_running(indexing_status):
                status = "not_ready"
            else:
                # Fallback to vector DB check for backward compatibility
                is_indexed = await db.has_file_indexed(thread_id, file_hash, thread.embed_model)
                status = "ready" if is_indexed else "not_ready"
        else:
            # Check all files in thread using file_status
            files = await get_thread_files(thread_id)
            if not files:
                status = "ready"
            else:
                all_indexed = True
                for f in files:
                    file_status = await get_file_status(f.file_hash)
                    scoped_indexing = get_scoped_indexing_status(
                        file_status,
                        embedding_model=thread.embed_model,
                        thread_id=thread_id,
                    )
                    indexing_status = scoped_indexing.get("status", ProcessStatus.UNKNOWN.value)
                    if not ProcessStatus.is_completed(indexing_status):
                        # Fallback to vector DB check for backward compatibility
                        if not await db.has_file_indexed(thread_id, f.file_hash, thread.embed_model):
                            all_indexed = False
                            break
                status = "ready" if all_indexed else "not_ready"

        stats = await db.get_thread_stats(
            thread_id=thread_id,
            file_hashes=[f.file_hash for f in files] if not file_hash else [file_hash],
            embedding_model_name=thread.embed_model,
        )

        # If vectors are missing and embedding model is offline, surface as blocked
        if status != "ready" and not embed_model_ready:
            status = "blocked"

        return {
            "thread_id": thread_id,
            "status": status,
            "stats": stats,
            "embed_model_ready": embed_model_ready,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
