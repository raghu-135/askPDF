import asyncio
import hashlib
import json
import logging
import os
import traceback
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.agent.agent import (
    build_system_prompt,
    get_tool_catalog,
    normalize_tool_instructions,
)
from app.db.database import (
    MessageRole,
    add_file_to_thread,
    count_threads_with_file_for_model,
    create_message,
    create_or_get_file,
    create_thread,
    delete_message,
    delete_message_pair,
    delete_thread,
    get_file,
    get_file_parsed_sentences,
    get_message,
    get_recent_messages,
    get_thread,
    get_thread_files,
    get_thread_file_annotations,
    get_thread_messages,
    get_thread_settings,
    get_thread_shape,
    is_file_in_thread,
    list_threads,
    recompute_qa_stats,
    remove_document_from_stats,
    remove_file_from_thread,
    update_file_parsed_sentences,
    update_thread,
    update_thread_settings,
    upsert_thread_stats_document,
    upsert_thread_file_annotations,
)
from app.db.vector_db import get_vector_db
from app.models.llm_server_client import (
    DEFAULT_EMBEDDING_MODEL,
    INTENT_AGENT_MAX_ITERATIONS,
    check_chat_model_ready,
    check_embed_model_ready,
    check_model_supports_tools,
    fetch_available_models,
    merge_thread_settings,
)
from app.models.requests import (
    PdfParseRequest,
    PromptDefaults,
    PromptPreviewRequest,
    RefreshWebSourceRequest,
    ThreadChatRequest,
    ThreadCreateRequest,
    ThreadFileRequest,
    ThreadFileAnnotationsResponse,
    ThreadFileAnnotationsUpdateRequest,
    ThreadSettingsResponse,
    ThreadSettingsUpdateRequest,
    ThreadUpdateRequest,
    ToolCatalogEntry,
    WebSourceRequest,
)
from app.rag.chat_service import handle_thread_chat
from app.rag.indexer import (
    index_document_for_thread,
    trigger_reembed_for_missing_sources,
)
from app.services.nlp_service import split_into_sentences
from app.services.parsing_service import extract_text_with_coordinates
from app.web_capture import capture_webpage_as_pdf, get_webpage_pdf_by_url_hash

router = APIRouter()
logger = logging.getLogger(__name__)


# ============ Compute Endpoints (Heavy Processing) ============


@router.post("/parse-pdf")
async def parse_pdf_endpoint(req: PdfParseRequest):
    """
    Extract structured text items and spatial coordinates (bounding boxes) from a PDF.
    Downloads the file from the backend and performs high-fidelity parsing to enable
    accurate PDF highlighting and sentence-level indexing.

    New format: sentences with word-level bboxes instead of character-level char_map.
    """
    pdf_url = f"{req.backend_url}/{req.file_hash}.pdf"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(pdf_url, timeout=30.0)
            resp.raise_for_status()
            pdf_data = resp.content

        # extract_text_with_coordinates now returns sentences directly with word-level bboxes
        sentences = extract_text_with_coordinates(pdf_data, filename=req.file_name)

        # Ensure file record exists in database
        await create_or_get_file(req.file_hash, req.file_name)

        # Store sentences in SQLite with version field
        parsed_data = {
            "version": "1.0",
            "sentences": sentences
        }
        await update_file_parsed_sentences(req.file_hash, json.dumps(parsed_data))

        return {"file_hash": req.file_hash, "sentences": sentences}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")


@router.get("/files/{file_hash}/parsed-sentences")
async def get_file_parsed_sentences_endpoint(file_hash: str):
    """
    Retrieve parsed sentences for a file from SQLite.
    Returns the JSON object with version and sentences array.
    """
    try:
        parsed_data = await get_file_parsed_sentences(file_hash)
        if not parsed_data:
            raise HTTPException(status_code=404, detail="Parsed sentences not found")
        return parsed_data
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve parsed sentences: {str(e)}")


# ============ Thread Endpoints ============


@router.get("/prompt-tools")
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


@router.post("/prompt-preview")
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
    """
    Create a new chat thread.
    """
    try:
        embed_model = (req.embed_model or "").strip() or DEFAULT_EMBEDDING_MODEL
        if not embed_model:
            raise HTTPException(
                status_code=400,
                detail="embed_model is required (set DEFAULT_EMBEDDING_MODEL or pass embed_model).",
            )
        # Create thread in SQLite
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
    """
    List all threads with message and file counts.
    """
    try:
        threads = await list_threads()
        return {"threads": threads}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads/{thread_id}")
async def get_thread_endpoint(thread_id: str):
    """
    Get a specific thread by ID.
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        asyncio.create_task(
            trigger_reembed_for_missing_sources(
                thread_id=thread_id,
                embedding_model_name=thread.embed_model,
            )
        )
        files = await get_thread_files(thread_id)
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
        # Delete thread-scoped vector data first
        db = get_vector_db()
        await db.delete_thread_data(thread_id)

        # Delete from SQLite
        deleted = await delete_thread(thread_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Thread not found")

        return {"status": "deleted", "thread_id": thread_id}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ Thread File Endpoints ============


@router.post("/threads/{thread_id}/files")
async def add_file_to_thread_endpoint(
    thread_id: str, req: ThreadFileRequest, background_tasks: BackgroundTasks
):
    """
    Add a file to a thread and trigger background indexing.
    """
    try:
        # Verify thread exists
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Create or get file record
        file = await create_or_get_file(req.file_hash, req.file_name, req.file_path)

        # Associate file with thread
        await add_file_to_thread(thread_id, req.file_hash)

        # Seed stats row with pending status immediately
        await upsert_thread_stats_document(
            thread_id=thread_id,
            file_hash=req.file_hash,
            file_name=req.file_name,
            source_type="pdf",
        )

        # Trigger background indexing
        background_tasks.add_task(
            index_document_for_thread,
            thread_id=thread_id,
            file_hash=req.file_hash,
            embedding_model_name=thread.embed_model,
        )

        return {
            "status": "accepted",
            "thread_id": thread_id,
            "file_hash": req.file_hash,
            "file_name": req.file_name,
            "indexing": "in_progress",
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads/{thread_id}/files")
async def get_thread_files_endpoint(thread_id: str):
    """
    Get all files associated with a thread.
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        files = await get_thread_files(thread_id)
        return {
            "thread_id": thread_id,
            "files": [
                {
                    "file_hash": f.file_hash,
                    "file_name": f.file_name,
                    "file_path": f.file_path,
                    "source_type": f.source_type,
                }
                for f in files
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads/{thread_id}/files/{file_hash}/annotations")
async def get_thread_file_annotations_endpoint(thread_id: str, file_hash: str):
    """
    Get the persisted annotation snapshot for a thread/file pair.
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        if not await is_file_in_thread(thread_id, file_hash):
            raise HTTPException(status_code=404, detail="File is not attached to this thread")

        row = await get_thread_file_annotations(thread_id, file_hash)
        if not row:
            return ThreadFileAnnotationsResponse(
                thread_id=thread_id,
                file_hash=file_hash,
                annotations=[],
            ).dict()

        return ThreadFileAnnotationsResponse(
            thread_id=thread_id,
            file_hash=file_hash,
            annotations=row["annotations"],
            created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else row["created_at"],
            updated_at=row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else row["updated_at"],
        ).dict()
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/threads/{thread_id}/files/{file_hash}/annotations")
async def update_thread_file_annotations_endpoint(
    thread_id: str,
    file_hash: str,
    req: ThreadFileAnnotationsUpdateRequest,
):
    """
    Replace the persisted annotation snapshot for a thread/file pair.
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        if not await is_file_in_thread(thread_id, file_hash):
            raise HTTPException(status_code=404, detail="File is not attached to this thread")

        row = await upsert_thread_file_annotations(thread_id, file_hash, req.annotations)
        return ThreadFileAnnotationsResponse(
            thread_id=thread_id,
            file_hash=file_hash,
            annotations=row["annotations"],
            created_at=row["created_at"].isoformat() if hasattr(row["created_at"], "isoformat") else row["created_at"],
            updated_at=row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else row["updated_at"],
        ).dict()
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threads/{thread_id}/web-sources")
async def add_web_source_to_thread_endpoint(
    thread_id: str,
    req: WebSourceRequest,
    background_tasks: BackgroundTasks,
):
    """
    Add a webpage URL to a thread: capture as PDF, record in DB, and trigger background indexing.

    Uses the unified PDF flow - web sources are converted to PDFs and processed
    identically to uploaded PDFs, enabling full annotation support.
    """
    import hashlib

    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        if not req.url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400, detail="URL must start with http:// or https://"
            )

        # Capture webpage and convert to PDF
        capture = await capture_webpage_as_pdf(req.url, force=False)

        # Use the PDF hash as the file identifier
        pdf_hash = capture["file_hash"]
        url_hash = capture["url_hash"]

        # Record in files table with source_type='pdf' (unified with uploaded PDFs)
        await create_or_get_file(
            file_hash=pdf_hash,
            file_name=capture["title"],
            file_path=capture["pdf_path"],
            source_type="pdf",  # Unified type
        )
        await add_file_to_thread(thread_id, pdf_hash)

        # Seed stats row with pending status immediately
        await upsert_thread_stats_document(
            thread_id=thread_id,
            file_hash=pdf_hash,
            file_name=capture["title"],
            source_type="pdf",
        )

        # Use markdown for vector DB indexing (cleaner than PDF extraction)
        background_tasks.add_task(
            index_document_for_thread,
            thread_id=thread_id,
            file_hash=pdf_hash,
            embedding_model_name=thread.embed_model,
            metadata={"original_url": req.url, "url_hash": url_hash},
            markdown_content=capture.get("markdown_content"),
        )

        return {
            "status": "accepted",
            "thread_id": thread_id,
            "file_hash": pdf_hash,
            "url_hash": url_hash,
            "url": req.url,
            "title": capture["title"],
            "source_type": "pdf",
            "indexing": "in_progress",
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/threads/{thread_id}/files/{file_hash}")
async def remove_source_from_thread_endpoint(thread_id: str, file_hash: str):
    """
    Remove a PDF or web source from a thread.
    Deletes vectors from Weaviate and removes the file-thread association from the DB.
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Delete vectors from Weaviate
        vector_db = get_vector_db()
        remaining_refs = await count_threads_with_file_for_model(
            file_hash=file_hash,
            embed_model=thread.embed_model,
            exclude_thread_id=thread_id,
        )
        if remaining_refs == 0:
            await vector_db.delete_source_chunks_by_file_hash(
                thread_id=thread_id,
                file_hash=file_hash,
                embedding_model_name=thread.embed_model,
            )

        # Remove from SQLite
        removed = await remove_file_from_thread(thread_id, file_hash)

        # Remove from thread stats
        await remove_document_from_stats(thread_id, file_hash)

        return {
            "status": "deleted",
            "thread_id": thread_id,
            "file_hash": file_hash,
            "removed_from_db": removed,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threads/{thread_id}/web-sources/{url_hash}/refresh")
async def refresh_web_source_endpoint(
    thread_id: str,
    url_hash: str,
    req: RefreshWebSourceRequest,
    background_tasks: BackgroundTasks,
):
    """
    Refresh a web source by recapturing it as a new PDF.

    With the unified PDF flow, refresh means:
    1. Look up the URL from the mapping file using url_hash
    2. Remove the old PDF from the thread (uses existing remove flow)
    3. Recapture the URL to a new PDF
    4. Add the new PDF to the thread (uses existing add flow)

    This is atomic and clean - no partial states.
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Look up the URL from the mapping file
        mapping = await get_webpage_pdf_by_url_hash(url_hash)
        if not mapping:
            raise HTTPException(status_code=404, detail="Web source mapping not found. The page may not have been captured yet.")

        url = mapping["url"]
        old_pdf_hash = mapping["pdf_hash"]

        # ── Phase 1: content-change check (compare PDF content hashes) ──────
        if req.content_hash and not req.confirmed:
            # Read stored content_hash from thread_stats for the old PDF
            shape = await get_thread_shape(thread_id)
            stored_meta = shape["documents"].get(old_pdf_hash, {})
            stored_hash = stored_meta.get("content_hash")

            if stored_hash and stored_hash == req.content_hash:
                return {
                    "status": "unchanged",
                    "message": "Page content has not changed since last index. No re-indexing needed.",
                    "thread_id": thread_id,
                    "url_hash": url_hash,
                    "file_hash": old_pdf_hash,
                }

            # Content changed → ask for confirmation
            return {
                "status": "confirmation_required",
                "message": "Page content has changed. Re-indexing will remove the current indexed data and replace it with the new content.",
                "thread_id": thread_id,
                "url_hash": url_hash,
                "file_hash": old_pdf_hash,
                "new_content_hash": req.content_hash,
            }

        # ── Phase 2: confirmed — remove old, recapture, add new ─────────────

        # 1. Remove old PDF from thread (reuse existing endpoint logic)
        await remove_source_from_thread_endpoint(thread_id, old_pdf_hash)

        # 2. Recapture to new PDF
        new_capture = await capture_webpage_as_pdf(url, force=True)
        new_pdf_hash = new_capture["file_hash"]

        # 3. Add new PDF to thread (reuse existing endpoint logic)
        add_result = await add_web_source_to_thread_endpoint(
            thread_id=thread_id,
            req=WebSourceRequest(url=url),
            background_tasks=background_tasks,
        )

        # 4. Clean up old PDF file if no longer referenced
        remaining_refs = await count_threads_with_file_for_model(
            file_hash=old_pdf_hash,
            embed_model=thread.embed_model,
        )
        if remaining_refs == 0:
            try:
                old_pdf_path = mapping["pdf_path"]
                if os.path.exists(old_pdf_path):
                    os.remove(old_pdf_path)
                    logger.info(f"Cleaned up old PDF: {old_pdf_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up old PDF {old_pdf_hash}: {e}")

        return {
            "status": "refreshed",
            "thread_id": thread_id,
            "url_hash": url_hash,
            "old_file_hash": old_pdf_hash,
            "new_file_hash": new_pdf_hash,
            "url": url,
            "title": new_capture["title"],
            "indexing": "in_progress",
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ Message Endpoints ============


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages_endpoint(
    thread_id: str, limit: int = 100, offset: int = 0
):
    """
    Get messages for a thread with pagination.
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        messages = await get_thread_messages(thread_id, limit, offset)
        return {
            "thread_id": thread_id,
            "messages": [
                {
                    "id": m.id,
                    "role": m.role.value,
                    "content": m.content,
                    "context_compact": m.context_compact,
                    "reasoning": m.reasoning,
                    "reasoning_available": m.reasoning_available,
                    "reasoning_format": m.reasoning_format,
                    "web_sources": m.web_sources,
                    "created_at": m.created_at.isoformat(),
                }
                for m in messages
            ],
            "limit": limit,
            "offset": offset,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/messages/{message_id}")
async def delete_message_endpoint(message_id: str):
    """
    Delete a message and its associated chat memory from Weaviate.
    If it's part of a QA pair, deletes both messages, their chat-memory vector,
    and any web search chunks (web_search type) whose URLs are no longer referenced
    by any other message in the thread.
    """
    try:
        # Get message to find thread_id and role
        message = await get_message(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        # ── Identify both sides of the QA pair ──
        all_msgs = await get_thread_messages(message.thread_id, limit=10000)
        assistant_msg_id = None
        if message.role == MessageRole.ASSISTANT:
            assistant_msg_id = message_id
        else:
            # USER → find the immediately following assistant message
            for i, m in enumerate(all_msgs):
                if (
                    m.id == message_id
                    and i + 1 < len(all_msgs)
                    and all_msgs[i + 1].role == MessageRole.ASSISTANT
                ):
                    assistant_msg_id = all_msgs[i + 1].id
                    break

        # IDs that will be removed from SQLite (this + its pair counterpart)
        ids_to_delete: set = {message_id}
        if assistant_msg_id and assistant_msg_id != message_id:
            ids_to_delete.add(assistant_msg_id)

        # ── Collect web_source URLs from the assistant message being deleted ──
        urls_to_check: set = set()
        if assistant_msg_id:
            asst_msg = await get_message(assistant_msg_id)
            if asst_msg and asst_msg.web_sources:
                for ws in asst_msg.web_sources:
                    url = ws.get("url", "").strip()
                    if url:
                        urls_to_check.add(url)

        db = get_vector_db()

        # ── Delete chat-memory vector ──
        vector_message_id = assistant_msg_id or message_id
        await db.delete_chat_memory_by_message_id(message.thread_id, vector_message_id)

        # ── Delete orphaned web_search chunks ──
        if urls_to_check:
            # URLs still referenced by other (surviving) messages
            still_needed: set = set()
            for m in all_msgs:
                if m.id not in ids_to_delete and m.web_sources:
                    for ws in m.web_sources:
                        url = ws.get("url", "").strip()
                        if url:
                            still_needed.add(url)
            orphaned = urls_to_check - still_needed
            if orphaned:
                await db.delete_web_chunks_by_urls(message.thread_id, list(orphaned))

        # ── Delete from SQLite (pair-aware) ──
        deleted_ids = await delete_message_pair(message_id)

        # ── Recompute QA stats to reflect the deletion ──
        try:
            await recompute_qa_stats(message.thread_id)
        except Exception as stats_err:
            logger.warning(f"thread_stats recompute skipped after delete: {stats_err}")

        return {"status": "deleted", "deleted_ids": deleted_ids}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ Chat Endpoints ============


@router.post("/threads/{thread_id}/chat")
async def thread_chat_endpoint(thread_id: str, req: ThreadChatRequest):
    """
    Thread-based chat with semantic memory.
    Returns answer, used_chat_ids (recollected messages), and document_sources.
    """
    try:
        # Verify thread exists
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Override thread_id from path
        req.thread_id = thread_id
        thread_settings = merge_thread_settings(await get_thread_settings(thread_id))
        if req.max_iterations is None:
            req.max_iterations = thread_settings["max_iterations"]
        if req.system_role_override is None:
            req.system_role_override = thread_settings["system_role"]
        if req.tool_instructions_override is None:
            req.tool_instructions_override = normalize_tool_instructions(
                thread_settings.get("tool_instructions", {})
            )
        if req.custom_instructions_override is None:
            req.custom_instructions_override = thread_settings["custom_instructions"]
        if req.use_intent_agent is None:
            req.use_intent_agent = thread_settings.get("use_intent_agent", True)
        if req.intent_agent_max_iterations is None:
            req.intent_agent_max_iterations = thread_settings.get(
                "intent_agent_max_iterations", INTENT_AGENT_MAX_ITERATIONS
            )
        if req.reasoning_mode is None:
            req.reasoning_mode = thread_settings.get("reasoning_mode", True)

        result = await handle_thread_chat(thread_id, req, thread.embed_model)
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ Indexing Endpoints ============


@router.get("/threads/{thread_id}/index-status")
async def get_thread_index_status(thread_id: str, file_hash: Optional[str] = None):
    """
    Check indexing status for a thread (or specific file in thread).
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        db = get_vector_db()
        embed_model_ready = await check_embed_model_ready(thread.embed_model)

        if file_hash:
            # Check specific file
            is_indexed = await db.has_file_indexed(thread_id, file_hash, thread.embed_model)
            status = "ready" if is_indexed else "not_ready"
        else:
            # Check all files in thread
            files = await get_thread_files(thread_id)
            if not files:
                status = "ready"
            else:
                all_indexed = True
                for f in files:
                    # Check if file chunks exist for this thread
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


# ============ Model Endpoints ============


@router.get("/models")
async def get_models():
    """
    Fetch available LLM and embedding models from the LLM API/server.
    """
    return await fetch_available_models()


@router.get("/health/is_chat_model_ready")
async def is_chat_model_ready_endpoint(model: str):
    """
    Check if a chat/LLM model is ready AND supports tool calling.

    Runs sequentially: tool-support is only probed when the model is confirmed
    ready, avoiding a wasted request to an unavailable model.
    Returns chat_model_ready and supports_tools as separate flags.
    """
    ready = await check_chat_model_ready(model)
    if not ready:
        return {"model": model, "chat_model_ready": False, "supports_tools": False}
    supports_tools = await check_model_supports_tools(model)
    return {"model": model, "chat_model_ready": True, "supports_tools": supports_tools}


@router.get("/health/is_embed_model_ready")
async def is_embed_model_ready_endpoint(model: str):
    """
    Check if an embedding model is ready.
    """
    ready = await check_embed_model_ready(model)
    return {"model": model, "embed_model_ready": ready}
