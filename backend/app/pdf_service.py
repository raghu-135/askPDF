

"""
PDFService: High-level business logic for PDF upload, extraction, and RAG indexing.

This service handles:
- Saving uploaded PDF files
- Extracting text and character coordinates
- Splitting text into sentences
- Mapping sentences to bounding boxes
- Triggering background RAG indexing
- Tracking indexing status per file

Dependencies:
- fastapi.HTTPException for error handling
- httpx for async HTTP requests
- pdf_parser and nlp modules for PDF/text processing
"""

import os
import hashlib
import httpx
import logging
import json
from enum import Enum
from typing import Dict, Optional, List
from datetime import datetime
from fastapi import HTTPException
from .pdf_parser import extract_text_with_coordinates
from .nlp import split_into_sentences


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
    Service class for handling PDF uploads, extraction, and RAG indexing.
    """
    def __init__(self, static_dir="/static", rag_service_url=None):
        """
        Initialize the PDFService.

        Args:
            static_dir (str): Directory to save uploaded PDFs and cached data.
            rag_service_url (str): URL for the RAG service. If not provided, uses RAG_SERVICE_URL env var.
        Raises:
            RuntimeError: If RAG_SERVICE_URL is not set.
        """
        self.static_dir = static_dir
        self.cache_dir = os.path.join(static_dir, "cache")
        self.rag_service_url = rag_service_url or os.getenv("RAG_SERVICE_URL")
        if not self.rag_service_url:
            logging.error("RAG_SERVICE_URL is not set. Please set the environment variable or pass it explicitly.")
            raise RuntimeError("RAG_SERVICE_URL is not set. Please set the environment variable or pass it explicitly.")
        os.makedirs(self.static_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    async def process_upload(self, file, embedding_model, background_tasks):
        """
        Handle the upload of a PDF file, extract text and bounding boxes, and trigger RAG indexing.

        Args:
            file (UploadFile): The uploaded PDF file.
            embedding_model (str): The embedding model to use for RAG.
            background_tasks (BackgroundTasks): FastAPI background tasks for async RAG call.
        Returns:
            dict: Contains sentences with bounding boxes, PDF URL, and file hash.
        Raises:
            HTTPException: If file is not a PDF or embedding_model is missing.
        """
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Please upload a PDF file.")
        if not embedding_model:
            raise HTTPException(status_code=400, detail="Please provide an embedding_model.")

        content = await file.read()
        file_hash = hashlib.md5(content).hexdigest()

        # Use file_hash as the PDF filename for consistency
        pdf_filename = f"{file_hash}.pdf"
        pdf_path = os.path.join(self.static_dir, pdf_filename)

        # Only write the file if it doesn't already exist (deduplication)
        if not os.path.exists(pdf_path):
            with open(pdf_path, "wb") as f:
                f.write(content)

        text, char_map = extract_text_with_coordinates(content)
        sentences = split_into_sentences(text)

        enriched_sentences = self._map_sentences_to_bboxes(sentences, text, char_map)

        # Save to cache
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.json")
        with open(cache_path, "w") as f:
            json.dump(enriched_sentences, f)

        # Initialize indexing status as pending
        set_indexing_status(file_hash, IndexingStatus.PENDING)

        # Trigger RAG Indexing in background
        background_tasks.add_task(
            self._call_rag,
            {"filename": file.filename, "file_hash": file_hash},
            embedding_model,
            file_hash
        )

        return {
            "sentences": enriched_sentences,
            "pdfUrl": f"/{pdf_filename}",
            "fileHash": file_hash,
            "indexingStatus": IndexingStatus.PENDING.value
        }

    @staticmethod
    def _map_sentences_to_bboxes(sentences, text, char_map):
        """
        Map each sentence to its bounding boxes by matching text in the extracted PDF.

        Args:
            sentences (list): List of sentence dicts with 'text' key.
            text (str): The full extracted text from the PDF.
            char_map (list): List of character coordinate dicts.
        Returns:
            list: Sentences with 'bboxes' key added for each.
        """
        current_idx = 0
        enriched_sentences = []
        for s in sentences:
            s_text = s["text"]
            s_text_clean = "".join(s_text.split())
            if not s_text_clean:
                s["bboxes"] = []
                enriched_sentences.append(s)
                continue
            match_start = -1
            match_end = -1
            s_ptr = 0
            temp_idx = current_idx
            while temp_idx < len(text) and s_ptr < len(s_text_clean):
                char = text[temp_idx]
                if char.isspace():
                    temp_idx += 1
                    continue
                if char == s_text_clean[s_ptr]:
                    if match_start == -1:
                        match_start = temp_idx
                    s_ptr += 1
                    temp_idx += 1
                else:
                    if match_start != -1:
                        temp_idx = match_start + 1
                        match_start = -1
                        s_ptr = 0
                    else:
                        temp_idx += 1
            if match_start != -1 and s_ptr == len(s_text_clean):
                match_end = temp_idx
                bboxes = char_map[match_start:match_end]
                s["bboxes"] = bboxes
                current_idx = match_end
            else:
                s["bboxes"] = []
            enriched_sentences.append(s)
        return enriched_sentences

    async def _call_rag(self, metadata: dict, emb_model: str, file_hash: str):
        """
        Asynchronously call the RAG service to index the PDF.
        The RAG service will download the PDF using the file_hash and parse it independently.

        Args:
            metadata (dict): Metadata about the PDF/file.
            emb_model (str): The embedding model to use.
            file_hash (str): The file hash for status tracking and PDF retrieval.
        """
        set_indexing_status(file_hash, IndexingStatus.INDEXING, progress=10)
        
        # Prepare payload - RAG service uses file_hash from metadata to fetch the PDF
        payload = {
            "embedding_model": emb_model,
            "metadata": metadata
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.rag_service_url}/index",
                    json=payload,
                    timeout=600.0  # 10 minutes for large PDFs
                )
                
                if response.status_code == 200:
                    set_indexing_status(file_hash, IndexingStatus.READY, progress=100)
                    print(f"RAG Indexing completed for {file_hash}", flush=True)
                else:
                    error_msg = f"RAG service returned status {response.status_code}"
                    set_indexing_status(file_hash, IndexingStatus.FAILED, error=error_msg)
                    print(f"RAG Indexing failed: {error_msg}", flush=True)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_msg = repr(e)
                set_indexing_status(file_hash, IndexingStatus.FAILED, error=error_msg)
                print(f"RAG Indexing failed: {error_msg}", flush=True)

    async def get_pdf_by_hash(self, file_hash: str) -> dict:
        """
        Get PDF data (sentences with bounding boxes) for an existing PDF by file hash.
        Uses cached data if available, otherwise re-parses the PDF.

        Args:
            file_hash (str): The MD5 hash of the PDF file.
        Returns:
            dict: Contains sentences with bounding boxes and PDF URL.
        Raises:
            HTTPException: If PDF file is not found.
        """
        pdf_filename = f"{file_hash}.pdf"
        pdf_path = os.path.join(self.static_dir, pdf_filename)
        cache_path = os.path.join(self.cache_dir, f"{file_hash}.json")

        if not os.path.exists(pdf_path):
            raise HTTPException(status_code=404, detail=f"PDF file not found: {file_hash}")

        # Try to load from cache first
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    enriched_sentences = json.load(f)
                return {
                    "sentences": enriched_sentences,
                    "pdfUrl": f"/{pdf_filename}",
                    "fileHash": file_hash
                }
            except Exception as e:
                logging.warning(f"Failed to load cache for {file_hash}: {e}")

        # Fallback: Read and re-parse the PDF content
        with open(pdf_path, "rb") as f:
            content = f.read()

        # Extract text and coordinates
        text, char_map = extract_text_with_coordinates(content)
        sentences = split_into_sentences(text)
        enriched_sentences = self._map_sentences_to_bboxes(sentences, text, char_map)

        # Update cache
        try:
            with open(cache_path, "w") as f:
                json.dump(enriched_sentences, f)
        except Exception as e:
            logging.warning(f"Failed to update cache for {file_hash}: {e}")

        return {
            "sentences": enriched_sentences,
            "pdfUrl": f"/{pdf_filename}",
            "fileHash": file_hash
        }
