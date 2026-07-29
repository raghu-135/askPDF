"""
Files API Module - File management endpoints.

Endpoints:
- POST /api/threads/{thread_id}/files/upload - Upload PDF
- POST /api/threads/{thread_id}/files - Add file to thread
- GET /api/threads/{thread_id}/files - List thread files
- GET /api/threads/{thread_id}/files/{file_hash} - Get PDF data
- GET /api/threads/{thread_id}/files/{file_hash}/download - Download PDF
- GET /api/threads/{thread_id}/files/{file_hash}/sentences - Get parsed sentences
- GET /api/threads/{thread_id}/files/{file_hash}/status - Get file status
- DELETE /api/threads/{thread_id}/files/{file_hash} - Remove file from thread
- GET /api/threads/{thread_id}/files/{file_hash}/annotations - Get annotations
- PUT /api/threads/{thread_id}/files/{file_hash}/annotations - Update annotations
- POST /api/threads/{thread_id}/browser-capture - Capture current browser page
"""

import hashlib
import os
import shutil
import traceback
from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from app.db import (
    DEFAULT_SENTENCES_JSON,
    EmbeddingReadinessStatus,
    FileSourceType,
    FileStatusSection,
    ProcessStatus,
    get_file,
    get_file_parsed_sentences,
    get_file_status,
    get_thread,
    get_thread_files,
    get_effective_thread_files,
    get_project,
    get_project_files,
    get_scoped_indexing_status,
    is_file_in_thread,
    is_file_in_project,
    is_file_accessible_to_thread,
    is_file_in_project_thread,
    add_file_to_project,
    remove_file_from_project,
    get_thread_file_annotations,
    remove_file_from_thread,
    update_parsing_status,
    upsert_thread_file_annotations,
)
from app.models.requests import (
    ThreadFileAnnotationsResponse,
    ThreadFileAnnotationsUpdateRequest,
    ThreadFileRequest,
)
from app.services.file_processing_service import (
    _default_file_status,
    _scoped_status_payload,
    queue_file_processing,
    queue_project_file_processing,
)
from app.services.file_cleanup_service import cleanup_detached_file
from app.time_utils import iso_utc_z, maybe_iso_utc_z
from app.services.embedding_model_service import (
    EmbeddingModelResolutionError,
    EmbeddingModelUnavailableError,
    require_thread_embedding_ready,
    require_embedding_model_ready,
)
from app.models.llm_server_client import check_embedding_model_ready

router = APIRouter(tags=["files"])

INDEXING_IN_PROGRESS = "in_progress"


async def _require_ready_thread(thread_id: str):
    try:
        return (await require_thread_embedding_ready(thread_id)).thread
    except EmbeddingModelResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "embedding_model_unavailable", "message": str(exc)},
        ) from exc


async def _require_ready_project(project_id: str):
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        await require_embedding_model_ready(project.embedding_model)
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "embedding_model_unavailable", "message": str(exc)},
        ) from exc
    return project


def _combined_processing_status(status: Dict[str, Any]) -> tuple[str, Optional[str]]:
    parsing = status.get("parsing") or {}
    indexing = status.get("indexing") or {}
    sections = (parsing, indexing)
    failed = next((section for section in sections if ProcessStatus.is_failed(section.get("status"))), None)
    if failed:
        return ProcessStatus.FAILED.value, str(failed.get("error") or "Processing failed")
    if all(ProcessStatus.is_completed(section.get("status")) for section in sections):
        return ProcessStatus.COMPLETED.value, None
    return ProcessStatus.PENDING.value, None


def _file_payload(file, *, scope: str):
    return {
        "file_hash": file.file_hash,
        "file_name": file.file_name,
        "file_path": file.file_path,
        "source_type": file.source_type,
        "association_scope": getattr(file, "association_scope", scope),
        "is_project_knowledge": getattr(file, "is_project_knowledge", scope == "project"),
        "added_at": maybe_iso_utc_z(getattr(file, "added_at", None)),
    }


async def _capture_current_page() -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{CAPTURE_SERVICE_URL}/capture", timeout=60.0)
        response.raise_for_status()
        capture = response.json()
    capture_path = f"/captures/{capture['file_hash']}.pdf"
    static_path = f"/static/{capture['file_hash']}.pdf"
    if not os.path.exists(static_path):
        if not os.path.exists(capture_path):
            raise HTTPException(status_code=500, detail=f"Captured PDF not found at {capture_path}")
        shutil.copy(capture_path, static_path)
    return capture


@router.post("/threads/{thread_id}/files/upload")
async def upload_pdf_endpoint(
    thread_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a PDF file, save it to static storage, and trigger background parsing and indexing.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    # Verify thread exists
    thread = await _require_ready_thread(thread_id)

    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()

    pdf_filename = f"{file_hash}.pdf"
    pdf_path = f"/static/{pdf_filename}"

    # Save PDF to static directory if not already exists
    import os
    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(content)

    # Queue file processing (parsing and indexing in background)
    await queue_file_processing(
        background_tasks=background_tasks,
        thread=thread,
        file_hash=file_hash,
        file_name=file.filename,
        backend_url="",  # Not needed - we serve files directly now
    )

    # Return immediately with sentences: null to indicate parsing not yet done
    return {
        "sentences": None,
        "download_url": f"/threads/{thread_id}/files/{file_hash}/download",
        "file_hash": file_hash,
        "file_name": file.filename,
    }


@router.post("/threads/{thread_id}/files")
async def add_file_to_thread_endpoint(
    thread_id: str, req: ThreadFileRequest, background_tasks: BackgroundTasks
):
    """Add a file to a thread and trigger background indexing."""
    try:
        # Verify thread exists
        thread = await _require_ready_thread(thread_id)

        await queue_file_processing(
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
            "indexing": INDEXING_IN_PROGRESS,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads/{thread_id}/files")
async def get_thread_files_endpoint(thread_id: str):
    """Get all files associated with a thread."""
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        files = await get_effective_thread_files(thread_id)
        payloads = []
        for file in files:
            direct = file.association_scope == "thread"
            status = _scoped_status_payload(
                file.file_hash,
                await get_file_status(file.file_hash),
                thread.embedding_model,
                thread_id if direct else None,
            )
            processing_status, processing_error = _combined_processing_status(status)
            payloads.append({
                **_file_payload(file, scope="thread"),
                "processing_status": processing_status,
                "processing_error": processing_error,
            })
        return {
            "thread_id": thread_id,
            "files": payloads,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threads/{thread_id}/files/{file_hash}")
async def get_pdf_data_endpoint(thread_id: str, file_hash: str):
    """
    Get PDF data (sentences with bounding boxes) for an existing PDF by file hash.
    """
    # Verify thread exists
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Verify file is attached to thread
    if not await is_file_accessible_to_thread(thread_id, file_hash):
        raise HTTPException(status_code=404, detail="File is not attached to this thread")

    import os
    pdf_path = f"/static/{file_hash}.pdf"

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file not found: {file_hash}")

    # Retrieve parsed sentences from database
    parsed_data = await get_file_parsed_sentences(file_hash)
    if parsed_data:
        sentences = parsed_data.get("sentences", [])
        return {
            "sentences": sentences,
            "download_url": f"/threads/{thread_id}/files/{file_hash}/download",
            "file_hash": file_hash,
        }

    # If not parsed yet, return empty sentences
    return {
        "sentences": [],
        "download_url": f"/threads/{thread_id}/files/{file_hash}/download",
        "file_hash": file_hash,
    }


@router.get("/threads/{thread_id}/files/{file_hash}/download")
async def download_pdf_endpoint(thread_id: str, file_hash: str):
    """
    Serve the actual PDF file from the static directory with CORS headers.
    Validates that the file is attached to the thread.
    """
    # Verify thread exists
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Verify file is attached to thread
    if not await is_file_accessible_to_thread(thread_id, file_hash):
        raise HTTPException(status_code=404, detail="File is not attached to this thread")

    import os
    file_path = f"/static/{file_hash}.pdf"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(file_path, media_type="application/pdf")


@router.head("/threads/{thread_id}/files/{file_hash}/download")
async def check_pdf_exists_endpoint(thread_id: str, file_hash: str):
    """
    Lightweight check to verify PDF is ready for download.
    Returns 200 if file exists and is attached to thread, 404 otherwise.
    """
    # Verify thread exists
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    # Verify file is attached to thread
    if not await is_file_accessible_to_thread(thread_id, file_hash):
        raise HTTPException(status_code=404, detail="File is not attached to this thread")

    file_path = f"/static/{file_hash}.pdf"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return Response(status_code=200)


@router.get("/threads/{thread_id}/files/{file_hash}/sentences")
async def get_file_parsed_sentences_endpoint(thread_id: str, file_hash: str):
    """
    Retrieve parsed sentences for a file from database.
    Returns the JSON object with version and sentences array.
    """
    try:
        # Verify thread exists
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Verify file is attached to thread
        if not await is_file_accessible_to_thread(thread_id, file_hash):
            # Return empty sentences instead of 404 - file may still be processing
            return DEFAULT_SENTENCES_JSON

        parsed_data = await get_file_parsed_sentences(file_hash)
        # Return data even if sentences is null (parsing pending) - never 404
        if parsed_data is None:
            # File exists but no parsing record yet - return default (matches DB init)
            return DEFAULT_SENTENCES_JSON
        return parsed_data
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve parsed sentences: {str(e)}")


@router.get("/threads/{thread_id}/files/{file_hash}/status")
async def get_file_status_endpoint(
    thread_id: str,
    file_hash: str,
    section: Optional[str] = None,
):
    """
    Retrieve file status (parsing and indexing status) from database.
    """
    try:
        # Verify thread exists
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Verify file is attached to thread
        if not await is_file_accessible_to_thread(thread_id, file_hash):
            # Check if file exists and is being processed
            file = await get_file(file_hash)
            if file:
                # File exists but not yet attached - return processing status
                # This handles the race condition where processing starts before attachment is committed
                status = _scoped_status_payload(
                    file_hash=file_hash,
                    status=await get_file_status(file_hash),
                    embedding_model=thread.embedding_model,
                    thread_id=thread_id,
                )
                # Override status to indicate processing
                status["parsing"] = {"status": ProcessStatus.PENDING.value}
                status["indexing"] = {"status": ProcessStatus.PENDING.value}
                return status
            else:
                raise HTTPException(status_code=404, detail="File is not attached to this thread")

        file = await get_file(file_hash)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")

        embedding_model = thread.embedding_model
        direct_association = await is_file_in_thread(thread_id, file_hash)

        status = _scoped_status_payload(
            file_hash=file_hash,
            status=await get_file_status(file_hash),
            embedding_model=embedding_model,
            thread_id=thread_id if direct_association else None,
        )
        parsing_status = (status.get("parsing") or {}).get("status", ProcessStatus.UNKNOWN.value)
        if not ProcessStatus.is_completed(parsing_status):
            parsed_data = await get_file_parsed_sentences(file_hash)
            sentences = parsed_data.get("sentences") if isinstance(parsed_data, dict) else None
            if isinstance(sentences, list) and sentences:
                await update_parsing_status(
                    file_hash,
                    ProcessStatus.COMPLETED.value,
                    finished_at=iso_utc_z(),
                )
                status = _scoped_status_payload(
                    file_hash=file_hash,
                    status=await get_file_status(file_hash),
                    embedding_model=embedding_model,
                    thread_id=thread_id if direct_association else None,
                )

        # Filter by section if specified
        if section:
            allowed_sections = {item.value for item in FileStatusSection}
            if section not in allowed_sections:
                raise HTTPException(status_code=400, detail=f"Invalid section: {section}")
            return {section: status[section]}

        return status
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file status: {str(e)}")


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

        if not await is_file_in_thread(thread_id, file_hash):
            raise HTTPException(status_code=404, detail="File is not directly attached to this thread")
        removed = await remove_file_from_thread(thread_id, file_hash)
        if removed:
            await cleanup_detached_file(file_hash, thread_id, thread.embedding_model)

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


@router.get("/threads/{thread_id}/files/{file_hash}/annotations")
async def get_thread_file_annotations_endpoint(thread_id: str, file_hash: str):
    """Get the persisted annotation snapshot for a thread/file pair."""
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        if not await is_file_accessible_to_thread(thread_id, file_hash):
            raise HTTPException(status_code=404, detail="File is not accessible to this thread")

        row = await get_thread_file_annotations(thread_id, file_hash)
        if not row:
            return ThreadFileAnnotationsResponse(
                thread_id=thread_id,
                file_hash=file_hash,
                annotations=[],
            ).model_dump()

        return ThreadFileAnnotationsResponse(
            thread_id=thread_id,
            file_hash=file_hash,
            annotations=row["annotations"],
            created_at=maybe_iso_utc_z(row["created_at"]),
            updated_at=maybe_iso_utc_z(row["updated_at"]),
        ).model_dump()
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
    """Replace the persisted annotation snapshot for a thread/file pair."""
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        if not await is_file_accessible_to_thread(thread_id, file_hash):
            raise HTTPException(status_code=404, detail="File is not accessible to this thread")

        row = await upsert_thread_file_annotations(thread_id, file_hash, req.annotations)
        return ThreadFileAnnotationsResponse(
            thread_id=thread_id,
            file_hash=file_hash,
            annotations=row["annotations"],
            created_at=maybe_iso_utc_z(row["created_at"]),
            updated_at=maybe_iso_utc_z(row["updated_at"]),
        ).model_dump()
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Browser capture service configuration
CAPTURE_SERVICE_URL = os.environ.get("CAPTURE_SERVICE_URL", "http://browser-capture:8080")


@router.post("/threads/{thread_id}/browser-capture")
async def capture_browser_page_endpoint(
    thread_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Capture the current browser page via browser-capture service,
    convert to PDF, and add to thread.
    """
    try:
        thread = await _require_ready_thread(thread_id)
        
        capture = await _capture_current_page()
        
        # Queue for processing (similar to web sources)
        await queue_file_processing(
            background_tasks=background_tasks,
            thread=thread,
            file_hash=capture["file_hash"],
            file_name=f"{capture['title']} - {capture['url']}",
            file_path=capture["url"],
            source_type=FileSourceType.BROWSER.value,
            indexing_metadata={
                "source_kind": "browser_capture",
                "url": capture["url"],
                "title": capture["title"],
            },
        )
        
        return {
            "status": EmbeddingReadinessStatus.READY.value,
            "thread_id": thread_id,
            "file_hash": capture["file_hash"],
            "url": capture["url"],
            "title": capture["title"],
            "indexing": INDEXING_IN_PROGRESS,
            "ready": True,
        }
        
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Browser capture service unavailable: {e}"
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/{project_id}/files/upload")
async def upload_project_pdf_endpoint(
    project_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    project = await _require_ready_project(project_id)
    content = await file.read()
    file_hash = hashlib.md5(content).hexdigest()
    pdf_path = f"/static/{file_hash}.pdf"
    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as output:
            output.write(content)
    await queue_project_file_processing(background_tasks, project, file_hash, file.filename)
    return {
        "sentences": None,
        "download_url": f"/projects/{project_id}/files/{file_hash}/download",
        "file_hash": file_hash,
        "file_name": file.filename,
    }


@router.post("/projects/{project_id}/files")
async def promote_file_to_project_endpoint(
    project_id: str,
    req: ThreadFileRequest,
    background_tasks: BackgroundTasks,
):
    project = await _require_ready_project(project_id)
    file = await get_file(req.file_hash)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    if (
        not await is_file_in_project(project_id, req.file_hash)
        and not await is_file_in_project_thread(project_id, req.file_hash)
    ):
        raise HTTPException(
            status_code=404,
            detail="File is not attached to a thread in this project",
        )
    await queue_project_file_processing(
        background_tasks,
        project,
        req.file_hash,
        req.file_name or file.file_name,
        req.file_path or file.file_path,
        file.source_type,
    )
    return {"status": "accepted", "project_id": project_id, "file_hash": req.file_hash}


@router.get("/projects/{project_id}/files")
async def get_project_files_endpoint(project_id: str, background_tasks: BackgroundTasks):
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    files = await get_project_files(project_id)
    if await check_embedding_model_ready(project.embedding_model):
        for file in files:
            await queue_project_file_processing(
                background_tasks,
                project,
                file.file_hash,
                file.file_name,
                file.file_path,
                file.source_type,
            )
    payloads = []
    for file in files:
        status = _scoped_status_payload(
            file.file_hash,
            await get_file_status(file.file_hash),
            project.embedding_model,
        )
        processing_status, processing_error = _combined_processing_status(status)
        payloads.append({
            **_file_payload(file, scope="project"),
            "processing_status": processing_status,
            "processing_error": processing_error,
        })
    return {"project_id": project_id, "files": payloads}


async def _require_project_file(project_id: str, file_hash: str):
    project = await get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    if not await is_file_in_project(project_id, file_hash):
        raise HTTPException(status_code=404, detail="File is not in project knowledge")
    file = await get_file(file_hash)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    return project, file


@router.get("/projects/{project_id}/files/{file_hash}")
async def get_project_pdf_data_endpoint(project_id: str, file_hash: str):
    await _require_project_file(project_id, file_hash)
    parsed_data = await get_file_parsed_sentences(file_hash) or {}
    return {
        "sentences": parsed_data.get("sentences") or [],
        "download_url": f"/projects/{project_id}/files/{file_hash}/download",
        "file_hash": file_hash,
    }


@router.get("/projects/{project_id}/files/{file_hash}/download")
async def download_project_pdf_endpoint(project_id: str, file_hash: str):
    await _require_project_file(project_id, file_hash)
    file_path = f"/static/{file_hash}.pdf"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(file_path, media_type="application/pdf")


@router.head("/projects/{project_id}/files/{file_hash}/download")
async def check_project_pdf_exists_endpoint(project_id: str, file_hash: str):
    await _require_project_file(project_id, file_hash)
    if not os.path.exists(f"/static/{file_hash}.pdf"):
        raise HTTPException(status_code=404, detail="PDF not found")
    return Response(status_code=200)


@router.get("/projects/{project_id}/files/{file_hash}/sentences")
async def get_project_sentences_endpoint(project_id: str, file_hash: str):
    await _require_project_file(project_id, file_hash)
    return await get_file_parsed_sentences(file_hash) or DEFAULT_SENTENCES_JSON


@router.get("/projects/{project_id}/files/{file_hash}/status")
async def get_project_file_status_endpoint(project_id: str, file_hash: str, section: Optional[str] = None):
    project, _ = await _require_project_file(project_id, file_hash)
    status = _scoped_status_payload(
        file_hash=file_hash,
        status=await get_file_status(file_hash),
        embedding_model=project.embedding_model,
    )
    if section:
        if section not in {item.value for item in FileStatusSection}:
            raise HTTPException(status_code=400, detail=f"Invalid section: {section}")
        return {section: status[section]}
    return status


@router.post("/projects/{project_id}/files/{file_hash}/retry")
async def retry_project_file_endpoint(project_id: str, file_hash: str, background_tasks: BackgroundTasks):
    project = await _require_ready_project(project_id)
    _, file = await _require_project_file(project_id, file_hash)
    await update_indexing_status(
        file_hash=file_hash,
        status=ProcessStatus.PENDING.value,
        embedding_model=project.embedding_model,
    )
    await queue_project_file_processing(
        background_tasks,
        project,
        file_hash,
        file.file_name,
        file.file_path,
        file.source_type,
    )
    return {"status": "accepted", "project_id": project_id, "file_hash": file_hash}


@router.post("/threads/{thread_id}/files/{file_hash}/retry")
async def retry_thread_file_endpoint(thread_id: str, file_hash: str, background_tasks: BackgroundTasks):
    context = await require_thread_embedding_ready(thread_id)
    if not await is_file_accessible_to_thread(thread_id, file_hash):
        raise HTTPException(status_code=404, detail="File is not available to this thread")
    file = await get_file(file_hash)
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    if await is_file_in_thread(thread_id, file_hash):
        await update_indexing_status(
            file_hash=file_hash,
            status=ProcessStatus.PENDING.value,
            embedding_model=context.embedding_model,
            thread_id=thread_id,
        )
        await queue_file_processing(
            background_tasks,
            context.thread,
            file_hash,
            file.file_name,
            file_path=file.file_path,
            source_type=file.source_type,
        )
    else:
        await update_indexing_status(
            file_hash=file_hash,
            status=ProcessStatus.PENDING.value,
            embedding_model=context.embedding_model,
        )
        await queue_project_file_processing(
            background_tasks,
            context.project,
            file_hash,
            file.file_name,
            file.file_path,
            file.source_type,
        )
    return {"status": "accepted", "thread_id": thread_id, "file_hash": file_hash}


@router.delete("/projects/{project_id}/files/{file_hash}")
async def remove_project_file_endpoint(project_id: str, file_hash: str):
    project, _ = await _require_project_file(project_id, file_hash)
    removed = await remove_file_from_project(project_id, file_hash)
    if removed:
        await cleanup_detached_file(file_hash, None, project.embedding_model)
    return {"status": "deleted", "project_id": project_id, "file_hash": file_hash}


@router.post("/projects/{project_id}/browser-capture")
async def capture_project_browser_page_endpoint(project_id: str, background_tasks: BackgroundTasks):
    try:
        project = await _require_ready_project(project_id)
        capture = await _capture_current_page()
        await queue_project_file_processing(
            background_tasks,
            project,
            capture["file_hash"],
            f"{capture['title']} - {capture['url']}",
            capture["url"],
            FileSourceType.BROWSER.value,
            {"source_kind": "browser_capture", "url": capture["url"], "title": capture["title"]},
        )
        return {
            "status": EmbeddingReadinessStatus.READY.value,
            "project_id": project_id,
            "file_hash": capture["file_hash"],
            "url": capture["url"],
            "title": capture["title"],
            "indexing": INDEXING_IN_PROGRESS,
            "ready": True,
        }
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"Browser capture service unavailable: {exc}") from exc
