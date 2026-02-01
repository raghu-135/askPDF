
from typing import List, Dict, Any
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from vectordb.base import VectorDBClient


class QdrantAdapter(VectorDBClient):
    """
    Adapter for Qdrant vector database, implementing the VectorDBClient interface.
    """
    def __init__(self) -> None:
        """Initialize the Qdrant client using environment variables for host and port."""
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        self.client = QdrantClient(host=host, port=port)


    async def index_documents(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> None:
        """
        Index documents into the specified Qdrant collection.
        Creates the collection if it does not exist.
        Uses UUIDs for point IDs.
        """
        if not embeddings:
            return

        dimension = len(embeddings[0])

        # Create collection if it does not exist
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text, **metadata}
            )
            for text, metadata, vector in zip(texts, metadatas, embeddings)
        ]

        # Batch upsert
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )


    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the specified collection.
        Returns a list of dicts with text, metadata, and score.
        """
        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit
        ).points

        results = []
        for hit in search_result:
            results.append({
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                "score": hit.score
            })
        return results


    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in Qdrant."""
        return self.client.collection_exists(collection_name)


    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from Qdrant."""
        self.client.delete_collection(collection_name=collection_name)
