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
)
from app.services.file_processing_service import (
    _default_file_status,
    _scoped_status_payload,
    queue_file_processing,
)
from app.services.file_cleanup_service import cleanup_detached_file

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
        "pdfUrl": f"/threads/{thread_id}/files/{file_hash}/download",
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

    # Retrieve parsed sentences from database
    parsed_data = await get_file_parsed_sentences(file_hash)
    if parsed_data:
        sentences = parsed_data.get("sentences", [])
        return {
            "sentences": sentences,
            "pdfUrl": f"/threads/{thread_id}/files/{file_hash}/download",
            "fileHash": file_hash,
        }

    # If not parsed yet, return empty sentences
    return {
        "sentences": [],
        "pdfUrl": f"/threads/{thread_id}/files/{file_hash}/download",
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
    if not await is_file_in_thread(thread_id, file_hash):
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
        if not await is_file_in_thread(thread_id, file_hash):
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
    embedding_model: Optional[str] = None,
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
        if not await is_file_in_thread(thread_id, file_hash):
            # Check if file exists and is being processed
            file = await get_file(file_hash)
            if file:
                # File exists but not yet attached - return processing status
                # This handles the race condition where processing starts before attachment is committed
                status = _scoped_status_payload(
                    file_hash=file_hash,
                    status=await get_file_status(file_hash),
                    embedding_model=thread.embed_model,
                    thread_id=thread_id,
                )
                # Override status to indicate processing
                status["parsing"] = {"status": "pending"}
                status["indexing"] = {"status": "pending"}
                return status
            else:
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

        # Remove from database
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
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Call browser-capture service
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{CAPTURE_SERVICE_URL}/capture",
                timeout=60.0
            )
            response.raise_for_status()
            capture = response.json()
        
        # Copy from shared volume /captures to /static for serving
        capture_path = f"/captures/{capture['file_hash']}.pdf"
        static_path = f"/static/{capture['file_hash']}.pdf"
        
        if not os.path.exists(static_path):
            if os.path.exists(capture_path):
                shutil.copy(capture_path, static_path)
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Captured PDF not found at {capture_path}"
                )
        
        # Queue for processing (similar to web sources)
        await queue_file_processing(
            background_tasks=background_tasks,
            thread=thread,
            file_hash=capture["file_hash"],
            file_name=f"{capture['title']} - {capture['url']}",
            file_path=capture["url"],
            source_type="browser",
            indexing_metadata={
                "source_kind": "browser_capture",
                "url": capture["url"],
                "title": capture["title"],
            },
        )
        
        return {
            "status": "ready",
            "thread_id": thread_id,
            "file_hash": capture["file_hash"],
            "url": capture["url"],
            "title": capture["title"],
            "indexing": "in_progress",
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
