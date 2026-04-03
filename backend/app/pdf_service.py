import os
import hashlib
import httpx
import logging
import json
from enum import Enum
from typing import Dict, Optional, List
from .service_client import ProcessingService
from datetime import datetime
from fastapi import HTTPException

# NOTE: Heavy PDF parsing and NLP logic moved to rag_service.
# We no longer import .pdf_parser or .nlp here.

class IndexingStatus(str, Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


class IndexingState:
    """Track the indexing state for a single file."""
    def __init__(self):
        self.status: IndexingStatus = IndexingStatus.PENDING
        self.error: Optional[str] = None
        self.started_at: datetime = datetime.now()
        self.finished_at: Optional[datetime] = None
        self.progress: int = 0  # Percentage 0-100

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "progress": self.progress
        }


# Global indexing status tracker (in production, use Redis or a database)
_indexing_status: Dict[str, IndexingState] = {}


def get_indexing_status(file_hash: str) -> Optional[IndexingState]:
    """Get the indexing status for a file."""
    return _indexing_status.get(file_hash)


def set_indexing_status(file_hash: str, status: IndexingStatus, error: Optional[str] = None, progress: int = 0):
    """Set the indexing status for a file."""
    if file_hash not in _indexing_status:
        _indexing_status[file_hash] = IndexingState()
    
    state = _indexing_status[file_hash]
    state.status = status
    state.progress = progress
    if error:
        state.error = error
    if status in (IndexingStatus.READY, IndexingStatus.FAILED):
        state.finished_at = datetime.now()


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
        os.makedirs(self.cache_dir, exist_ok=True)

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

    async def process_upload(self, file, embedding_model, background_tasks):
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")
        if not embedding_model:
            raise HTTPException(status_code=400, detail="Please provide an embedding_model.")

        content = await file.read()
        file_hash = hashlib.md5(content).hexdigest()

        pdf_filename = f"{file_hash}.pdf"
        pdf_path = os.path.join(self.static_dir, pdf_filename)

        if not os.path.exists(pdf_path):
            with open(pdf_path, "wb") as f:
                f.write(content)

        # Delegate parsing to RAG service
        enriched_sentences = await self._delegate_parsing(file_hash, file.filename)

        # Save to cache
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.json")
        with open(cache_path, "w") as f:
            json.dump(enriched_sentences, f)

        set_indexing_status(file_hash, IndexingStatus.PENDING)

        # Trigger RAG Indexing in background (calls separate /index endpoint in RAG service)
        background_tasks.add_task(
            self._call_rag_index,
            {"filename": file.filename, "file_hash": file_hash},
            embedding_model,
            file_hash
        )

        return {
            "sentences": enriched_sentences,
            "pdfUrl": f"/api/pdf-file/{file_hash}",
            "fileHash": file_hash,
            "indexingStatus": IndexingStatus.PENDING.value
        }

    async def _call_rag_index(self, metadata: dict, emb_model: str, file_hash: str):
        """
        Coordinate the indexing call to ensure documents are processed into the vector database.
        """
        set_indexing_status(file_hash, IndexingStatus.INDEXING, progress=10)
        
        try:
            success = await self.service_client.index_document(metadata, emb_model)
            if success:
                set_indexing_status(file_hash, IndexingStatus.READY, progress=100)
            else:
                set_indexing_status(file_hash, IndexingStatus.FAILED, error="Indexing service failed.")
        except Exception as e:
            set_indexing_status(file_hash, IndexingStatus.FAILED, error=str(e))

    async def get_pdf_by_hash(self, file_hash: str) -> dict:
        pdf_filename = f"{file_hash}.pdf"
        pdf_path = os.path.join(self.static_dir, pdf_filename)
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.json")

        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {file_hash}")

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

        # Fallback: re-parse via RAG service
        enriched_sentences = await self._delegate_parsing(file_hash, pdf_filename)

        try:
            with open(cache_path, "w") as f:
                json.dump(enriched_sentences, f)
        except Exception as e:
            logging.warning(f"Failed to update cache for {file_hash}: {e}")

        return {
            "sentences": enriched_sentences,
            "pdfUrl": f"/api/pdf-file/{file_hash}",
            "fileHash": file_hash
        }
