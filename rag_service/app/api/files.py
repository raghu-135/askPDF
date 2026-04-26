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
- POST /api/threads/{thread_id}/web-sources - Add web source
- POST /api/threads/{thread_id}/web-sources/{url_hash}/refresh - Refresh web source
"""

import hashlib
import traceback
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.db import (
    get_file,
    get_file_parsed_sentences,
    get_file_status,
    get_thread,
    get_thread_files,
    get_scoped_indexing_status,
    is_file_in_thread,
    get_thread_file_annotations,
    remove_file_from_thread,
    upsert_thread_file_annotations,
)
from app.models.requests import (
    ThreadFileAnnotationsResponse,
    ThreadFileAnnotationsUpdateRequest,
    ThreadFileRequest,
    WebSourceRequest,
    RefreshWebSourceRequest,
)
from app.services.file_processing_service import (
    _default_file_status,
    _scoped_status_payload,
    queue_file_processing,
)
from app.services.file_cleanup_service import cleanup_detached_file
from app.web_capture import capture_webpage_as_pdf, get_webpage_pdf_by_url_hash

router = APIRouter(tags=["files"])


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
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

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
        "pdfUrl": f"/api/threads/{thread_id}/files/{file_hash}/download",
        "fileHash": file_hash,
        "fileName": file.filename,
    }


@router.post("/threads/{thread_id}/files")
async def add_file_to_thread_endpoint(
    thread_id: str, req: ThreadFileRequest, background_tasks: BackgroundTasks
):
    """Add a file to a thread and trigger background indexing."""
    try:
        # Verify thread exists
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

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
            "indexing": "in_progress",
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
    if not await is_file_in_thread(thread_id, file_hash):
        raise HTTPException(status_code=404, detail="File is not attached to this thread")

    import os
    pdf_path = f"/static/{file_hash}.pdf"

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF file not found: {file_hash}")

    # Retrieve parsed sentences from SQLite
    parsed_data = await get_file_parsed_sentences(file_hash)
    if parsed_data:
        sentences = parsed_data.get("sentences", [])
        return {
            "sentences": sentences,
            "pdfUrl": f"/api/threads/{thread_id}/files/{file_hash}/download",
            "fileHash": file_hash,
        }

    # If not parsed yet, return empty sentences
    return {
        "sentences": [],
        "pdfUrl": f"/api/threads/{thread_id}/files/{file_hash}/download",
        "fileHash": file_hash,
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
    if not await is_file_in_thread(thread_id, file_hash):
        raise HTTPException(status_code=404, detail="File is not attached to this thread")

    import os
    file_path = f"/static/{file_hash}.pdf"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(file_path, media_type="application/pdf")


@router.get("/threads/{thread_id}/files/{file_hash}/sentences")
async def get_file_parsed_sentences_endpoint(thread_id: str, file_hash: str):
    """
    Retrieve parsed sentences for a file from SQLite.
    Returns the JSON object with version and sentences array.
    """
    try:
        # Verify thread exists
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Verify file is attached to thread
        if not await is_file_in_thread(thread_id, file_hash):
            raise HTTPException(status_code=404, detail="File is not attached to this thread")

        parsed_data = await get_file_parsed_sentences(file_hash)
        if not parsed_data:
            raise HTTPException(status_code=404, detail="Parsed sentences not found")
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
    embedding_model: Optional[str] = None,
):
    """
    Retrieve file status (parsing and indexing status) from SQLite.
    """
    try:
        # Verify thread exists
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Verify file is attached to thread
        if not await is_file_in_thread(thread_id, file_hash):
            raise HTTPException(status_code=404, detail="File is not attached to this thread")

        file = await get_file(file_hash)
        if not file:
            raise HTTPException(status_code=404, detail="File not found")

        # Use thread's embed_model if not specified
        if not embedding_model:
            embedding_model = thread.embed_model

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
            await cleanup_detached_file(file_hash, thread_id, thread.embed_model)

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
    """Replace the persisted annotation snapshot for a thread/file pair."""
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
    """
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

        await queue_file_processing(
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


@router.post("/threads/{thread_id}/web-sources/{url_hash}/refresh")
async def refresh_web_source_endpoint(
    thread_id: str,
    url_hash: str,
    req: RefreshWebSourceRequest,
    background_tasks: BackgroundTasks,
):
    """
    Refresh a web source by recapturing it as a new PDF.
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

        # Content-change check (compare PDF content hashes)
        from app.db import get_thread_shape
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

        # Confirmed — remove old, recapture, add new
        # 1. Remove old PDF from thread
        await remove_source_from_thread_endpoint(thread_id, old_pdf_hash)

        # 2. Recapture to new PDF
        new_capture = await capture_webpage_as_pdf(url, force=True)
        new_pdf_hash = new_capture["file_hash"]

        # 3. Add new PDF to thread
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
