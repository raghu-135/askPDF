import os
import hashlib
import httpx
import logging
import json
from typing import Dict, Optional, List
from .service_client import ProcessingService
from fastapi import HTTPException

# NOTE: Heavy PDF parsing and NLP logic moved to rag_service.
# We no longer import .pdf_parser or .nlp here.
# NOTE: Indexing status tracking moved to RAG service database (file_status column).


class PDFService:
    """
    Service class for coordinate-aware PDF processing and RAG indexing coordination.
    Handles file upload management, caching of parsed results, and delegation of tasks to specialized services.
    """
    def __init__(self, static_dir="/static", service_client: ProcessingService = None):
        self.static_dir = static_dir
        self.cache_dir = os.path.join(static_dir, "cache")
        self.service_client = service_client
        # Internal URL for processing service to reach backend for PDF download
        self.backend_internal_url = os.getenv("BACKEND_INTERNAL_URL", "http://backend:8000")

        if not self.service_client:
            raise RuntimeError("ServiceClient is not provided.")

        os.makedirs(self.static_dir, exist_ok=True)
        # Cache directory is no longer created here - kept for backward compatibility only

    async def _delegate_parsing(self, file_hash: str, filename: str) -> List[dict]:
        """
        Request enriched PDF parsing from the processing service.
        Returns a list of sentences with bounding boxes and font metadata.
        """
        try:
            return await self.service_client.parse_pdf(file_hash, filename, self.backend_internal_url)
        except Exception as e:
            logging.error(f"Failed to delegate parsing: {e}")
            raise HTTPException(status_code=502, detail="PDF parsing service unavailable or failed.")

    async def process_upload(self, file, thread_id: str):
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")
        if not thread_id:
            raise HTTPException(status_code=400, detail="Please provide a thread_id.")

        content = await file.read()
        file_hash = hashlib.md5(content).hexdigest()

        pdf_filename = f"{file_hash}.pdf"
        pdf_path = os.path.join(self.static_dir, pdf_filename)

        if not os.path.exists(pdf_path):
            with open(pdf_path, "wb") as f:
                f.write(content)

        # Tell RAG service to process PDF (parse and index in background)
        await self.service_client.process_pdf(file_hash, file.filename, self.backend_internal_url, thread_id)

        # Return immediately with sentences: null to indicate parsing not yet done
        return {
            "sentences": None,
            "pdfUrl": f"/api/pdf-file/{file_hash}",
            "fileHash": file_hash,
            "fileName": file.filename
        }

    async def get_pdf_by_hash(self, file_hash: str) -> dict:
        pdf_filename = f"{file_hash}.pdf"
        pdf_path = os.path.join(self.static_dir, pdf_filename)
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.json")

        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {file_hash}")

        # Try to retrieve from RAG service SQLite first
        parsed_data = await self.service_client.get_parsed_sentences(file_hash)
        if parsed_data:
            sentences = parsed_data.get("sentences", [])
            return {
                "sentences": sentences,
                "pdfUrl": f"/api/pdf-file/{file_hash}",
                "fileHash": file_hash
            }

        # Fallback to file cache for existing files
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    enriched_sentences = json.load(f)
                return {
                    "sentences": enriched_sentences,
                    "pdfUrl": f"/api/pdf-file/{file_hash}",
                    "fileHash": file_hash
                }
            except Exception as e:
                logging.warning(f"Failed to load cache for {file_hash}: {e}")

        # Fallback: re-parse via RAG service (which will now store in SQLite)
        enriched_sentences = await self._delegate_parsing(file_hash, pdf_filename)

        return {
            "sentences": enriched_sentences,
            "pdfUrl": f"/api/pdf-file/{file_hash}",
            "fileHash": file_hash
        }
