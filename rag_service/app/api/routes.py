import asyncio
import hashlib
import json
import logging
import os
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse

from app.agent.agent import (
    build_system_prompt,
    get_tool_catalog,
    normalize_tool_instructions,
)
from app.db import (
    MessageRole,
    ProcessStatus,
    add_file_to_thread,
    count_threads_with_file,
    count_threads_with_file_for_model,
    create_message,
    create_or_get_file,
    create_thread,
    delete_file_record,
    delete_message,
    delete_message_pair,
    delete_thread,
    get_file,
    get_file_parsed_sentences,
    get_scoped_indexing_status,
    get_file_status,
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
    remove_thread_indexing_status,
    update_file_parsed_sentences,
    update_file_status,
    update_parsing_status,
    update_indexing_status,
    update_thread,
    update_thread_settings,
    upsert_document_in_stats,
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
    ProcessPdfRequest,
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
WEBPAGES_DIR = "/static/webpages"


def _default_file_status(file_hash: str) -> Dict[str, Any]:
    """Return the default status payload for an unknown file."""
    return {
        "file_hash": file_hash,
        "parsing": {"status": ProcessStatus.UNKNOWN.value},
        "indexing": {"status": ProcessStatus.UNKNOWN.value},
        "indexing_status": {
            "summary": {"status": ProcessStatus.UNKNOWN.value},
            "models": {},
        },
        "updated_at": None,
    }


def _scoped_status_payload(
    file_hash: str,
    status: Optional[Dict[str, Any]],
    embedding_model: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a backward-compatible file-status payload with a scoped top-level indexing section."""
    payload = dict(status or _default_file_status(file_hash))
    payload["file_hash"] = file_hash
    payload["indexing"] = get_scoped_indexing_status(
        payload,
        embedding_model=embedding_model,
        thread_id=thread_id,
    )
    return payload


async def _delete_file_artifacts(file_hash: str) -> None:
    """Delete the stored PDF and any webpage mapping that points to it."""
    pdf_path = f"/static/{file_hash}.pdf"
    if os.path.exists(pdf_path):
        try:
            os.remove(pdf_path)
        except Exception as exc:
            logger.warning("Failed to delete PDF artifact %s: %s", pdf_path, exc)

    if not os.path.isdir(WEBPAGES_DIR):
        return

    for filename in os.listdir(WEBPAGES_DIR):
        if not filename.endswith(".mapping.json"):
            continue
        mapping_path = os.path.join(WEBPAGES_DIR, filename)
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if mapping.get("pdf_hash") != file_hash:
                continue
            os.remove(mapping_path)
        except Exception as exc:
            logger.warning("Failed to delete webpage mapping %s: %s", mapping_path, exc)


def _get_web_mapping_by_pdf_hash(file_hash: str) -> Optional[Dict[str, Any]]:
    """Return the webpage mapping payload for a PDF hash when one exists."""
    if not os.path.isdir(WEBPAGES_DIR):
        return None

    for filename in os.listdir(WEBPAGES_DIR):
        if not filename.endswith(".mapping.json"):
            continue
        mapping_path = os.path.join(WEBPAGES_DIR, filename)
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if mapping.get("pdf_hash") == file_hash:
                return mapping
        except Exception as exc:
            logger.warning("Failed to read webpage mapping %s: %s", mapping_path, exc)
    return None


async def _repair_thread_documents_meta(thread_id: str, embedding_model: str, files: List[Any]) -> None:
    """Rebuild thread_stats.documents_meta for already-indexed files attached to a thread."""
    vector_db = get_vector_db()
    attached_hashes = {f.file_hash for f in files}
    shape = await get_thread_shape(thread_id)
    for stale_hash in list(shape.get("documents", {}).keys()):
        if stale_hash not in attached_hashes:
            await remove_document_from_stats(thread_id, stale_hash)

    for file in files:
        file_status = await get_file_status(file.file_hash)
        scoped_status = get_scoped_indexing_status(
            file_status,
            embedding_model=embedding_model,
            thread_id=thread_id,
        )
        chunk_count = await vector_db.get_file_chunk_count(file.file_hash, embedding_model)
        is_ready = ProcessStatus.is_completed(scoped_status.get("status", ProcessStatus.UNKNOWN.value)) or chunk_count > 0
        if not is_ready:
            await remove_document_from_stats(thread_id, file.file_hash)
            continue

        total_chars = int(scoped_status.get("total_chars", 0) or 0)
        mapping = _get_web_mapping_by_pdf_hash(file.file_hash) if file.source_type == "web" else None
        content_hash = None
        if mapping and mapping.get("markdown_content"):
            content_hash = hashlib.md5(mapping["markdown_content"].encode("utf-8")).hexdigest()
        await upsert_document_in_stats(
            thread_id,
            file.file_hash,
            {
                "file_name": file.file_name,
                "source_type": file.source_type,
                "chunk_count": chunk_count,
                "total_chars": total_chars,
                "indexing_status": ProcessStatus.COMPLETED.value,
                "indexed_at": scoped_status.get("finished_at"),
                **({"url": file.file_path} if file.source_type == "web" and file.file_path else {}),
                **({"title": mapping.get("title")} if mapping and mapping.get("title") else {}),
                **({"content_hash": content_hash} if content_hash else {}),
            },
        )


async def _cleanup_detached_file(file_hash: str, thread_id: str, embed_model: str) -> None:
    """Apply post-detach cleanup for status, vector data, and orphaned file artifacts."""
    await remove_document_from_stats(thread_id, file_hash)
    await remove_thread_indexing_status(file_hash, embed_model, thread_id)

    vector_db = get_vector_db()
    remaining_model_refs = await count_threads_with_file_for_model(file_hash, embed_model)
    if remaining_model_refs == 0:
        await vector_db.delete_document_vectors_by_file_hash_and_model(
            file_hash=file_hash,
            embedding_model_name=embed_model,
        )

    remaining_refs = await count_threads_with_file(file_hash)
    if remaining_refs == 0:
        file_status = await get_file_status(file_hash) or {}
        indexing_status = file_status.get("indexing_status", {})
        models = indexing_status.get("models", {}) if isinstance(indexing_status, dict) else {}
        model_names = [name for name in models.keys() if isinstance(name, str) and name]
        if not model_names:
            model_names = [embed_model]
        for model_name in model_names:
            await vector_db.delete_document_vectors_by_file_hash_and_model(
                file_hash=file_hash,
                embedding_model_name=model_name,
            )
        await delete_file_record(file_hash)
        await _delete_file_artifacts(file_hash)


async def _queue_file_processing(
    background_tasks: BackgroundTasks,
    thread,
    file_hash: str,
    file_name: str,
    backend_url: str = "",  # No longer needed, files are read locally
    file_path: Optional[str] = None,
    source_type: str = "pdf",
    indexing_metadata: Optional[Dict[str, Any]] = None,
    markdown_content: Optional[str] = None,
) -> None:
    """Ensure a file is attached to a thread and background parse/index work is queued."""
    await create_or_get_file(
        file_hash=file_hash,
        file_name=file_name,
        file_path=file_path,
        source_type=source_type,
    )
    await add_file_to_thread(thread.id, file_hash)

    file_status = await get_file_status(file_hash)
    parsing_status = (file_status or {}).get("parsing", {"status": ProcessStatus.UNKNOWN.value})

    scoped_indexing = get_scoped_indexing_status(
        file_status,
        embedding_model=thread.embed_model,
        thread_id=thread.id,
    )
    if not ProcessStatus.is_completed(scoped_indexing.get("status", ProcessStatus.UNKNOWN.value)) and not ProcessStatus.is_running(scoped_indexing.get("status", ProcessStatus.UNKNOWN.value)):
        await update_indexing_status(
            file_hash=file_hash,
            status=ProcessStatus.PENDING.value,
            embedding_model=thread.embed_model,
            thread_id=thread.id,
        )
        background_tasks.add_task(
            _background_index,
            file_hash,
            thread.id,
            thread.embed_model,
            file_name,
            backend_url,
            indexing_metadata or {},
            markdown_content,
        )

    parsed_data = await get_file_parsed_sentences(file_hash)
    if parsed_data:
        if not ProcessStatus.is_completed(parsing_status.get("status", ProcessStatus.UNKNOWN.value)):
            await update_parsing_status(file_hash, ProcessStatus.COMPLETED.value)
    elif not ProcessStatus.is_running(parsing_status.get("status", ProcessStatus.UNKNOWN.value)):
        await update_parsing_status(file_hash, ProcessStatus.PENDING.value)
        background_tasks.add_task(_background_parse, file_hash, file_name, backend_url)


# ============ Compute Endpoints (Heavy Processing) ============


@router.post("/process-pdf")
async def process_pdf_endpoint(req: ProcessPdfRequest, background_tasks: BackgroundTasks):
    """
    Process a PDF file (parse and index) in the background.
    This endpoint returns immediately after triggering the background tasks.
    """
    thread = await get_thread(req.thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    await _queue_file_processing(
        background_tasks=background_tasks,
        thread=thread,
        file_hash=req.file_hash,
        file_name=req.file_name,
        backend_url=req.backend_url,
    )

    return {
        "status": "accepted",
        "thread_id": req.thread_id,
        "file_hash": req.file_hash,
    }


async def _background_parse(file_hash: str, filename: str, backend_url: str = ""):
    """
    Background task to parse PDF and update status.
    Reads PDF from local disk at /static/{file_hash}.pdf
    """
    current_status = await get_file_status(file_hash)
    parsing_status = (current_status or {}).get("parsing", {"status": ProcessStatus.UNKNOWN.value})

    if ProcessStatus.is_completed(parsing_status.get("status", ProcessStatus.UNKNOWN.value)):
        if await get_file_parsed_sentences(file_hash):
            return

    started_at = datetime.utcnow().isoformat()
    try:
        claimed = await update_parsing_status(
            file_hash,
            ProcessStatus.RUNNING.value,
            started_at=started_at,
            claim=True,
        )
        if not claimed:
            return

        # Read PDF from local disk
        pdf_path = f"/static/{file_hash}.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

        sentences = extract_text_with_coordinates(pdf_data, filename=filename)
        parsed_data = {
            "version": "1.0",
            "sentences": sentences
        }
        await update_file_parsed_sentences(file_hash, json.dumps(parsed_data))

        finished_at = datetime.utcnow().isoformat()
        await update_parsing_status(
            file_hash,
            ProcessStatus.COMPLETED.value,
            started_at=started_at,
            finished_at=finished_at,
        )
        logger.info(f"Background parsing completed for {file_hash}")
    except Exception as e:
        traceback.print_exc()
        finished_at = datetime.utcnow().isoformat()
        try:
            await update_parsing_status(
                file_hash,
                ProcessStatus.FAILED.value,
                started_at=started_at,
                finished_at=finished_at,
                error=str(e),
            )
        except Exception as update_error:
            logger.error(f"Failed to update parsing status to failed for {file_hash}: {update_error}")
        logger.error(f"Background parsing failed for {file_hash}: {e}")


async def _background_index(
    file_hash: str,
    thread_id: str,
    embedding_model: str,
    file_name: str,
    backend_url: str,
    metadata: Optional[Dict[str, Any]] = None,
    markdown_content: Optional[str] = None,
):
    """
    Background task to index a document for a thread after parsing completes.
    """
    started_at = datetime.utcnow().isoformat()
    try:
        claimed = await update_indexing_status(
            file_hash=file_hash,
            status=ProcessStatus.RUNNING.value,
            embedding_model=embedding_model,
            thread_id=thread_id,
            started_at=started_at,
            claim=True,
        )
        if not claimed:
            return

        result = await index_document_for_thread(
            thread_id=thread_id,
            file_hash=file_hash,
            embedding_model_name=embedding_model,
            metadata=metadata,
            markdown_content=markdown_content,
        )
        if result.get("status") != "success":
            raise Exception(result.get("message", "Indexing failed"))
        logger.info(f"Background indexing completed for %s in thread %s", file_hash, thread_id)

        if markdown_content is None:
            await _background_parse(file_hash, file_name, backend_url)
    except Exception as e:
        traceback.print_exc()
        finished_at = datetime.utcnow().isoformat()
        try:
            await update_indexing_status(
                file_hash=file_hash,
                status=ProcessStatus.FAILED.value,
                embedding_model=embedding_model,
                thread_id=thread_id,
                started_at=started_at,
                finished_at=finished_at,
                error=str(e),
            )
        except Exception as update_error:
            logger.error(f"Failed to update indexing status to failed for {file_hash}: {update_error}")
        logger.error(f"Background indexing failed for {file_hash}: {e}")


@router.post("/parse-pdf")
async def parse_pdf_endpoint(req: PdfParseRequest):
    """
    Extract structured text items and spatial coordinates (bounding boxes) from a PDF.
    Reads the file from local disk at /static/{file_hash}.pdf and performs high-fidelity
    parsing to enable accurate PDF highlighting and sentence-level indexing.

    New format: sentences with word-level bboxes instead of character-level char_map.
    """
    # Update parsing status to running
    await update_parsing_status(req.file_hash, ProcessStatus.RUNNING.value, started_at=datetime.utcnow().isoformat())

    try:
        # Read PDF from local disk
        pdf_path = f"/static/{req.file_hash}.pdf"
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")

        with open(pdf_path, "rb") as f:
            pdf_data = f.read()

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

        # Update parsing status to completed
        await update_parsing_status(req.file_hash, ProcessStatus.COMPLETED.value, finished_at=datetime.utcnow().isoformat())

        return {"file_hash": req.file_hash, "sentences": sentences}
    except Exception as e:
        traceback.print_exc()
        # Update parsing status to failed
        await update_parsing_status(req.file_hash, ProcessStatus.FAILED.value, error=str(e))
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


@router.get("/files/{file_hash}/status")
async def get_file_status_endpoint(
    file_hash: str,
    section: Optional[str] = None,
    embedding_model: Optional[str] = None,
    thread_id: Optional[str] = None,
):
    """
    Retrieve file status (parsing and indexing status) from SQLite.
    Returns the file_status JSON with parsing and indexing sections.
    
    Query parameters:
    - section: Optional filter for specific section (e.g., "parsing", "indexing")
    """
    try:
        file = await get_file(file_hash)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        status = _scoped_status_payload(
            file_hash=file_hash,
            status=await get_file_status(file_hash),
            embedding_model=embedding_model,
            thread_id=thread_id,
        )

        # Filter by section if specified
        if section:
            if section not in {"parsing", "indexing"}:
                raise HTTPException(status_code=400, detail=f"Invalid section: {section}")
            return {section: status[section]}

        return status
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file status: {str(e)}")


@router.put("/files/{file_hash}/status")
async def update_file_status_endpoint(file_hash: str, status_update: Dict[str, Any]):
    """
    Update file status (parsing and indexing status) in SQLite.
    Accepts partial status updates and merges with existing status.
    """
    try:
        file = await get_file(file_hash)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")
        
        success = await update_file_status(file_hash, status_update)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update file status")
        
        return await get_file_status_endpoint(file_hash)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to update file status: {str(e)}")


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

        files = await get_thread_files(thread_id)
        await _repair_thread_documents_meta(thread_id, thread.embed_model, files)
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

        # Delete from SQLite
        deleted = await delete_thread(thread_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Thread not found")

        for file in files:
            await _cleanup_detached_file(file.file_hash, thread_id, thread.embed_model)

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

        await _queue_file_processing(
            background_tasks=background_tasks,
            thread=thread,
            file_hash=req.file_hash,
            file_name=req.file_name,
            file_path=req.file_path,
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

        markdown_content = capture.get("markdown_content") or ""
        content_hash = hashlib.md5(markdown_content.encode("utf-8")).hexdigest() if markdown_content else None

        await _queue_file_processing(
            background_tasks=background_tasks,
            thread=thread,
            file_hash=pdf_hash,
            file_name=capture["title"],
            file_path=req.url,
            source_type="web",
            indexing_metadata={
                "source_kind": "webpage",
                "url": req.url,
                "title": capture["title"],
                "url_hash": url_hash,
                **({"content_hash": content_hash} if content_hash else {}),
            },
            markdown_content=markdown_content,
        )

        return {
            "status": "accepted",
            "thread_id": thread_id,
            "file_hash": pdf_hash,
            "url_hash": url_hash,
            "url": req.url,
            "title": capture["title"],
            "source_type": "web",
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

        # Remove from SQLite
        removed = await remove_file_from_thread(thread_id, file_hash)
        if removed:
            await _cleanup_detached_file(file_hash, thread_id, thread.embed_model)

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


# ============ File Upload and Serving Endpoints (Migrated from Backend) ============


@router.post("/upload")
async def upload_pdf_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    thread_id: str = Form(...),
):
    """
    Upload a PDF file, save it to static storage, and trigger background parsing and indexing.

    Args:
        file (UploadFile): PDF file to upload.
        thread_id (str): Thread to attach the upload to immediately.
    Returns:
        dict: Result with fileHash, fileName, pdfUrl, and sentences (null initially).
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    if not thread_id:
        raise HTTPException(status_code=400, detail="Please provide a thread_id.")

    # Verify thread exists
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()

    pdf_filename = f"{file_hash}.pdf"
    pdf_path = f"/static/{pdf_filename}"

    # Save PDF to static directory if not already exists
    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(content)

    # Queue file processing (parsing and indexing in background)
    await _queue_file_processing(
        background_tasks=background_tasks,
        thread=thread,
        file_hash=file_hash,
        file_name=file.filename,
        backend_url="",  # Not needed - we serve files directly now
    )

    # Return immediately with sentences: null to indicate parsing not yet done
    return {
        "sentences": None,
        "pdfUrl": f"/files/{file_hash}.pdf",
        "fileHash": file_hash,
        "fileName": file.filename,
    }


@router.get("/files/{file_hash}.pdf")
async def get_pdf_file_endpoint(file_hash: str):
    """
    Serve the actual PDF file from the static directory with CORS headers.
    """
    file_path = f"/static/{file_hash}.pdf"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(file_path, media_type="application/pdf")


@router.get("/files/{file_hash}")
async def get_pdf_data_endpoint(file_hash: str):
    """
    Get PDF data (sentences with bounding boxes) for an existing PDF by file hash.
    This is used to reload PDFs when switching threads.

    Args:
        file_hash (str): The MD5 hash of the PDF file.
    Returns:
        dict: Contains sentences with bounding boxes and PDF URL.
    Raises:
        HTTPException: If PDF file is not found.
    """
    pdf_path = f"/static/{file_hash}.pdf"

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file not found: {file_hash}")

    # Retrieve parsed sentences from SQLite
    parsed_data = await get_file_parsed_sentences(file_hash)
    if parsed_data:
        sentences = parsed_data.get("sentences", [])
        return {
            "sentences": sentences,
            "pdfUrl": f"/files/{file_hash}.pdf",
            "fileHash": file_hash,
        }

    # If not parsed yet, return empty sentences
    return {
        "sentences": [],
        "pdfUrl": f"/files/{file_hash}.pdf",
        "fileHash": file_hash,
    }
