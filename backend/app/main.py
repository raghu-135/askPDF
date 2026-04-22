"""
main.py
-------
FastAPI entry point for the AskPDF backend service.

This service acts as a coordinator and API gateway for:
- PDF uploads and metadata management
- Coordination of PDF parsing and indexing tasks (delegated to the processing service)
- Managing indexing status and serving processed assets
"""
import os
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Local imports
from .pdf_service import PDFService



API_PREFIX = "/api"
# External service configuration
PROCESSING_SERVICE_URL = os.getenv("PROCESSING_SERVICE_URL")
if PROCESSING_SERVICE_URL is None:
    raise ValueError("PROCESSING_SERVICE_URL environment variable is not set")


app = FastAPI(title="PDF TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
os.makedirs("/static", exist_ok=True)



from .service_client import ProcessingService, RestProcessingServiceClient
service_client: ProcessingService = RestProcessingServiceClient(service_url=PROCESSING_SERVICE_URL)
pdf_service = PDFService(static_dir="/static", service_client=service_client)

@app.post(f"{API_PREFIX}/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    thread_id: str = Form(...),
):
    """
    Upload a PDF file, extract sentences and bounding boxes, and trigger RAG indexing.

    Args:
        file (UploadFile): PDF file to upload.
        thread_id (str): Thread to attach the upload to immediately.
    Returns:
        dict: Result of the upload and processing.
    """
    return await pdf_service.process_upload(file, thread_id)


@app.get(f"{API_PREFIX}/pdf/{{file_hash}}")
async def get_pdf_data(file_hash: str):
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
    return await pdf_service.get_pdf_by_hash(file_hash)


@app.get(f"{API_PREFIX}/pdf-file/{{file_hash}}")
async def get_pdf_file(file_hash: str):
    """
    Serve the actual PDF file from the static directory with CORS headers.
    This replaces serving it via StaticFiles which lacks CORS on some setups.
    """
    file_path = f"/static/{file_hash}.pdf"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(file_path, media_type="application/pdf")


@app.get(f"{API_PREFIX}/files/{{file_hash}}/status")
async def get_file_status_endpoint(
    file_hash: str,
    section: Optional[str] = None,
    embedding_model: Optional[str] = None,
    thread_id: Optional[str] = None,
):
    """
    Check the file status (parsing and indexing) for a specific file.
    
    Args:
        file_hash (str): The MD5 hash of the file to check status for.
    Returns:
        dict: File status information with parsing and indexing sections.
    """
    status = await service_client.get_file_status(
        file_hash,
        section=section,
        embedding_model=embedding_model,
        thread_id=thread_id,
    )
    if status is None:
        # Unknown file - could be already indexed or never uploaded
        payload = {
            "file_hash": file_hash,
            "parsing": {"status": "unknown"},
            "indexing": {"status": "unknown"},
            "updated_at": None
        }
        if section == "parsing":
            return {"parsing": payload["parsing"]}
        if section == "indexing":
            return {"indexing": payload["indexing"]}
        return payload
    return status


@app.get("/health")
async def health():
    """
    Service health check endpoint.
    Returns:
        Status OK if service is running.
    """
    return {"status": "ok"}


# Mount static files last to avoid shadowing API routes
app.mount("/", StaticFiles(directory="/static", html=True), name="static")
