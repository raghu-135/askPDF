from typing import List, Dict, Any
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from vectordb.base import VectorDBClient

class QdrantAdapter(VectorDBClient):
    def __init__(self):
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        self.client = QdrantClient(host=host, port=port)

    async def index_documents(self, collection_name: str, texts: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
        if not embeddings:
            return

        dimension = len(embeddings[0])
        
        # Check if collection exists, if not create
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )

        points = [
            models.PointStruct(
                id=idx, # Simple integer ID, relying on consistent indexing order for this demo. Ideally use UUIDs.
                vector=vector,
                payload={"text": text, **metadata}
            )
            for idx, (text, metadata, vector) in enumerate(zip(texts, metadatas, embeddings))
        ]
        
        # Batch upsert
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )

    async def search(self, collection_name: str, query_vector: List[float], limit: int = 5) -> List[Dict[str, Any]]:
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
        return self.client.collection_exists(collection_name)

    async def delete_collection(self, collection_name: str):
        self.client.delete_collection(collection_name=collection_name)
