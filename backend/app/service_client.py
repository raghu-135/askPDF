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
    async def index_document(self, metadata: dict, emb_model: str) -> bool:
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

    async def index_document(self, metadata: dict, emb_model: str) -> bool:
        """
        Trigger document indexing in the processing service.
        """
        payload = {
            "embedding_model": emb_model,
            "metadata": metadata
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.service_url}/index",
                    json=payload,
                    timeout=600.0
                )
                return response.status_code == 200
            except httpx.HTTPError as e:
                logger.error(f"Failed to index document: {e}")
                return False

