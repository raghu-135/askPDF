import os
import httpx
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)

class ProcessingService(ABC):
    """
    Interface defining the contract for the core processing service.
    Implementations of this interface should handle communication with specialized
    services for PDF parsing and indexing.
    """
    @abstractmethod
    async def parse_pdf(self, file_hash: str, filename: str, backend_url: str) -> List[dict]:
        pass

    @abstractmethod
    async def get_parsed_sentences(self, file_hash: str) -> Optional[dict]:
        pass

    @abstractmethod
    async def get_file_status(
        self,
        file_hash: str,
        section: Optional[str] = None,
        embedding_model: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Optional[dict]:
        pass

    @abstractmethod
    async def process_pdf(self, file_hash: str, filename: str, backend_url: str, thread_id: str) -> bool:
        pass

class RestProcessingServiceClient(ProcessingService):
    """
    REST API implementation of the core processing service client.
    Handles communication with external REST-based services (formerly RAG Service).
    """
    def __init__(self, service_url: Optional[str] = None):
        self.service_url = service_url or os.getenv("PROCESSING_SERVICE_URL")
        if not self.service_url:
            logger.error("PROCESSING_SERVICE_URL is not set.")
            raise RuntimeError("PROCESSING_SERVICE_URL is not configured.")

    async def parse_pdf(self, file_hash: str, filename: str, backend_url: str) -> List[dict]:
        """
        Request enriched PDF parsing from the processing service.
        Returns a list of sentences with bounding boxes and font metadata.
        """
        payload = {
            "file_hash": file_hash,
            "file_name": filename,
            "backend_url": backend_url
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.service_url}/parse-pdf",
                    json=payload,
                    timeout=300.0
                )
                response.raise_for_status()
                data = response.json()
                return data.get("sentences", [])
            except httpx.HTTPError as e:
                logger.error(f"Failed to parse PDF: {e}")
                raise

    async def get_parsed_sentences(self, file_hash: str) -> Optional[dict]:
        """
        Retrieve parsed sentences for a file from the processing service.
        Returns the JSON object with version and sentences array, or None if not found.
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.service_url}/files/{file_hash}/parsed-sentences",
                    timeout=30.0
                )
                if response.status_code == 200:
                    return response.json()
                return None
            except httpx.HTTPError as e:
                logger.error(f"Failed to retrieve parsed sentences: {e}")
                return None

    async def get_file_status(
        self,
        file_hash: str,
        section: Optional[str] = None,
        embedding_model: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Retrieve file status (parsing and indexing status) from the processing service.
        Returns the file_status JSON with parsing and indexing sections, or None if not found.
        """
        params = {}
        if section:
            params["section"] = section
        if embedding_model:
            params["embedding_model"] = embedding_model
        if thread_id:
            params["thread_id"] = thread_id
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{self.service_url}/files/{file_hash}/status",
                    params=params or None,
                    timeout=30.0
                )
                if response.status_code == 200:
                    return response.json()
                return None
            except httpx.HTTPError as e:
                logger.error(f"Failed to retrieve file status: {e}")
                return None

    async def process_pdf(self, file_hash: str, filename: str, backend_url: str, thread_id: str):
        """
        Tell the processing service to process a PDF file (parse and index).
        The service will handle background parsing and indexing with status updates internally.
        """
        payload = {
            "thread_id": thread_id,
            "file_hash": file_hash,
            "file_name": filename,
            "backend_url": backend_url,
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.service_url}/process-pdf",
                    json=payload,
                    timeout=30.0
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                logger.error(f"Failed to process PDF: {e}")
                raise
