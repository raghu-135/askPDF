from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorDBClient(ABC):
    """Abstract base class for Vector Database interactions."""

    @abstractmethod
    async def index_documents(self, collection_name: str, texts: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
        """Index documents into the vector database."""
        pass

    @abstractmethod
    async def search(self, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents using a query vector."""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str):
        """Delete a collection."""
        pass
