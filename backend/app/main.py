"""
main.py
-------
FastAPI entry point for the AskPDF backend service.

This service acts as a coordinator and API gateway for:
- PDF uploads and metadata management
- Coordination of PDF parsing and indexing tasks (delegated to the processing service)
- Proxying Text-to-Speech (TTS) and Web Capture requests to the processing service
- Managing indexing status and serving processed assets
"""
import os
import uuid
import hashlib
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# Local imports
from .tts import tts_sentence_to_wav
from .pdf_service import PDFService, get_indexing_status, IndexingStatus



API_PREFIX = "/api"
AUDIO_DIR = "/data/audio"
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

# Ensure audio and static directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs("/static", exist_ok=True)



from .service_client import ProcessingService, RestProcessingServiceClient
service_client: ProcessingService = RestProcessingServiceClient(service_url=PROCESSING_SERVICE_URL)
pdf_service = PDFService(static_dir="/static", service_client=service_client)

@app.post(f"{API_PREFIX}/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    embedding_model: str = Form(...)
):
    """
    Upload a PDF file, extract sentences and bounding boxes, and trigger RAG indexing.
    
    Args:
        background_tasks (BackgroundTasks): FastAPI background task manager.
        file (UploadFile): PDF file to upload.
        embedding_model (str): Name of the embedding model to use.
    Returns:
        dict: Result of the upload and processing.
    """
    return await pdf_service.process_upload(file, embedding_model, background_tasks)


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


@app.get(f"{API_PREFIX}/voices")
async def get_voices():
    """
    List available TTS voices/styles.
    
    Returns:
        dict: Available voices/styles for TTS.
    """
    voices = await service_client.list_voices()
    return {"voices": voices}


@app.post(f"{API_PREFIX}/tts")
async def synthesize_sentence(payload: dict):
    """
    Synthesize audio for a sentence using TTS.
    
    Args:
        payload (dict): Should contain 'text', 'voice', and optional 'speed'.
    Returns:
        dict: URL to the generated audio file.
    Raises:
        HTTPException: If 'text' is missing in the payload.
    """
    text = payload.get("text")
    voice = payload.get("voice")
    speed = payload.get("speed", 1.0)
    if not text:
        raise HTTPException(status_code=400, detail="Missing 'text' in payload.")
    path = await tts_sentence_to_wav(service_client, text, AUDIO_DIR, voice_style=voice, speed=speed)
    rel = os.path.relpath(path, "/")
    url = f"/{rel}"
    return {"audioUrl": url}


@app.get(f"{API_PREFIX}/index-status/{{file_hash}}")
async def get_file_index_status(file_hash: str):
    """
    Check the indexing status for a specific file.
    
    Args:
        file_hash (str): The MD5 hash of the file to check status for.
    Returns:
        dict: Status information including status, progress, and any errors.
    """
    status = get_indexing_status(file_hash)
    if status is None:
        # Unknown file - could be already indexed or never uploaded
        return {
            "file_hash": file_hash,
            "status": "unknown",
            "message": "File not found in indexing queue. It may already be indexed or was never uploaded."
        }
    return {
        "file_hash": file_hash,
        **status.to_dict()
    }


class WebCaptureRequest(BaseModel):
    url: str
    force: bool = False


@app.post(f"{API_PREFIX}/web-capture")
async def web_capture(req: WebCaptureRequest):
    """
    Proxy a webpage capture request to the processing service.
    The processing service performs the capture, inlines assets, and saves the 
    resulting self-contained HTML file to the shared volume (/static/webpages).
    """
    try:
        return await service_client.web_capture(url=req.url, force=req.force)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to capture page: {exc}")


@app.get(f"{API_PREFIX}/web-page/{{file_hash}}")
async def get_web_page(file_hash: str):
    """
    Return the saved self-contained HTML for a captured webpage.

    The frontend fetches this as text, wraps it in a Blob URL, and renders
    it inside an <iframe> — bypassing any X-Frame-Options / CSP from the
    original site entirely.
    """
    html_path = f"/static/webpages/{file_hash}.html"
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Web page capture not found.")
    return FileResponse(
        html_path,
        media_type="text/html",
        headers={
            # Explicitly allow framing from our own frontend origin
            "X-Frame-Options": "ALLOWALL",
            "Content-Security-Policy": "frame-ancestors *",
        },
    )


@app.get("/health")
async def health():
    """
    Service health check endpoint.
    Returns:
        Status OK if service is running.
    """
    return {"status": "ok"}


# Serve audio files
app.mount("/data", StaticFiles(directory="/data"), name="data")

# Mount static files last to avoid shadowing API routes
app.mount("/", StaticFiles(directory="/static", html=True), name="static")
