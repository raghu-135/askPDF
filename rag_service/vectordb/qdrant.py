"""
qdrant.py - Qdrant vector database adapter with per-thread collections

This adapter supports:
- Per-thread collections (col_thread_{thread_id})
- PDF chunk storage with file_hash filtering
- Chat QA memory storage and retrieval
- Semantic search across both PDF and chat content
"""

from typing import List, Dict, Any, Optional
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from vectordb.base import VectorDBClient


class QdrantAdapter(VectorDBClient):
    """
    Adapter for Qdrant vector database, implementing the VectorDBClient interface.
    Supports per-thread collections and dual-search (PDF + chat memory).
    """
    def __init__(self) -> None:
        """Initialize the Qdrant client using environment variables for host and port."""
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))
        self.client = QdrantClient(host=host, port=port)

    # ============ Collection Naming Helpers ============
    
    @staticmethod
    def get_thread_collection_name(thread_id: str) -> str:
        """Get the collection name for a thread."""
        return f"col_thread_{thread_id}"
    
    @staticmethod
    def get_legacy_collection_name(embedding_model_name: str, file_hash: Optional[str] = None) -> str:
        """
        Generate a safe collection name for legacy compatibility.
        Used for non-thread-based indexing (backward compatible).
        """
        base_model_name = embedding_model_name.split(":")[0]
        safe_model_name = base_model_name.replace("-", "_").replace(".", "_").replace("/", "_")
        if file_hash:
            return f"rag_{safe_model_name}_{file_hash}"
        return f"rag_{safe_model_name}"

    # ============ Collection Management ============

    async def create_thread_collection(self, thread_id: str, vector_size: int = 768) -> str:
        """
        Create a new collection for a thread.
        Returns the collection name.
        """
        collection_name = self.get_thread_collection_name(thread_id)
        
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, 
                    distance=models.Distance.COSINE
                ),
            )
            print(f"Created thread collection: {collection_name}", flush=True)
        
        return collection_name

    async def delete_thread_collection(self, thread_id: str) -> bool:
        """Delete a thread's collection."""
        collection_name = self.get_thread_collection_name(thread_id)
        try:
            if self.client.collection_exists(collection_name):
                self.client.delete_collection(collection_name=collection_name)
                print(f"Deleted thread collection: {collection_name}", flush=True)
                return True
            return False
        except Exception as e:
            print(f"Error deleting collection {collection_name}: {e}", flush=True)
            return False

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in Qdrant."""
        return self.client.collection_exists(collection_name)

    async def thread_collection_exists(self, thread_id: str) -> bool:
        """Check if a thread's collection exists."""
        collection_name = self.get_thread_collection_name(thread_id)
        return self.client.collection_exists(collection_name)

    async def delete_collection(self, collection_name: str) -> None:
        """Delete a collection from Qdrant."""
        self.client.delete_collection(collection_name=collection_name)

    # ============ PDF Chunk Operations ============

    async def index_pdf_chunks(
        self,
        thread_id: str,
        file_hash: str,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> int:
        """
        Index PDF chunks into a thread's collection.
        Returns the number of chunks indexed.
        """
        if not embeddings:
            return 0

        collection_name = self.get_thread_collection_name(thread_id)
        
        # Ensure collection exists with correct dimension
        if not self.client.collection_exists(collection_name):
            await self.create_thread_collection(thread_id, len(embeddings[0]))

        points = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            payload = {
                "text": text,
                "type": "pdf_chunk",
                "file_hash": file_hash,
                "chunk_id": i,
                **metadata
            }
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
            )

        # Batch upsert
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        print(f"Indexed {len(points)} PDF chunks for thread {thread_id}, file {file_hash}", flush=True)
        return len(points)

    async def search_pdf_chunks(
        self,
        thread_id: str,
        query_vector: List[float],
        limit: int = 5,
        file_hash: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for PDF chunks in a thread's collection.
        Optionally filter by file_hash.
        """
        collection_name = self.get_thread_collection_name(thread_id)
        
        if not self.client.collection_exists(collection_name):
            return []

        # Build filter for PDF chunks
        must_conditions = [
            models.FieldCondition(
                key="type",
                match=models.MatchValue(value="pdf_chunk")
            )
        ]
        
        if file_hash:
            must_conditions.append(
                models.FieldCondition(
                    key="file_hash",
                    match=models.MatchValue(value=file_hash)
                )
            )

        search_filter = models.Filter(must=must_conditions)

        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=limit
        ).points

        results = []
        for hit in search_result:
            results.append({
                "text": hit.payload.get("text", ""),
                "file_hash": hit.payload.get("file_hash"),
                "chunk_id": hit.payload.get("chunk_id"),
                "type": "pdf_chunk",
                "score": hit.score,
                "metadata": {k: v for k, v in hit.payload.items() if k not in ["text", "type"]}
            })
        return results

    async def get_chunks_by_ids(
        self,
        thread_id: str,
        file_hash: str,
        chunk_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """Retrieve specific PDF chunks by their chunk_ids."""
        collection_name = self.get_thread_collection_name(thread_id)
        if not self.client.collection_exists(collection_name):
            return []

        # Filter for the specific chunks
        must_conditions = [
            models.FieldCondition(key="type", match=models.MatchValue(value="pdf_chunk")),
            models.FieldCondition(key="file_hash", match=models.MatchValue(value=file_hash)),
        ]
        
        # Use MatchAny for multiple chunk IDs
        should_conditions = [
            models.FieldCondition(
                key="chunk_id",
                match=models.MatchAny(any=chunk_ids)
            )
        ]
        
        search_filter = models.Filter(
            must=must_conditions,
            should=should_conditions
        )

        # Scroll is better than query for fetching known points
        scroll_result, _ = self.client.scroll(
            collection_name=collection_name,
            scroll_filter=search_filter,
            limit=len(chunk_ids),
            with_payload=True,
            with_vectors=False
        )

        results = []
        for hit in scroll_result:
            results.append({
                "text": hit.payload.get("text", ""),
                "file_hash": hit.payload.get("file_hash"),
                "chunk_id": hit.payload.get("chunk_id"),
                "type": "pdf_chunk",
                "metadata": {k: v for k, v in hit.payload.items() if k not in ["text", "type"]}
            })
        
        # Sort by chunk_id to maintain sequence
        results.sort(key=lambda x: x["chunk_id"])
        return results

    # ============ Chat Memory Operations ============

    async def upsert_chat_memory(
        self,
        thread_id: str,
        message_id: str,
        question: str,
        answer: str,
        embedding: List[float]
    ) -> bool:
        """
        Store a QA pair as chat memory for semantic retrieval.
        The text is stored as "Q: ...\nA: ..." format.
        """
        collection_name = self.get_thread_collection_name(thread_id)
        
        if not self.client.collection_exists(collection_name):
            await self.create_thread_collection(thread_id, len(embedding))

        qa_text = f"Q: {question}\nA: {answer}"
        
        point = models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": qa_text,
                "type": "chat_memory",
                "message_id": message_id,
                "thread_id": thread_id,
                "question": question,
                "answer": answer
            }
        )

        self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )
        
        print(f"Stored chat memory for message {message_id} in thread {thread_id}", flush=True)
        return True

    async def search_chat_memory(
        self,
        thread_id: str,
        query_vector: List[float],
        limit: int = 3,
        exclude_message_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chat memory (past QA pairs) in a thread.
        Optionally exclude specific message IDs (e.g., the current conversation).
        """
        collection_name = self.get_thread_collection_name(thread_id)
        
        if not self.client.collection_exists(collection_name):
            return []

        # Build filter for chat memory
        must_conditions = [
            models.FieldCondition(
                key="type",
                match=models.MatchValue(value="chat_memory")
            )
        ]
        
        must_not_conditions = []
        if exclude_message_ids:
            for msg_id in exclude_message_ids:
                must_not_conditions.append(
                    models.FieldCondition(
                        key="message_id",
                        match=models.MatchValue(value=msg_id)
                    )
                )

        search_filter = models.Filter(
            must=must_conditions,
            must_not=must_not_conditions if must_not_conditions else None
        )

        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=limit
        ).points

        results = []
        for hit in search_result:
            results.append({
                "text": hit.payload.get("text", ""),
                "message_id": hit.payload.get("message_id"),
                "question": hit.payload.get("question"),
                "answer": hit.payload.get("answer"),
                "type": "chat_memory",
                "score": hit.score
            })
        return results

    async def delete_chat_memory_by_message_id(
        self,
        thread_id: str,
        message_id: str
    ) -> bool:
        """
        Delete chat memory associated with a specific message ID.
        Used when a message is deleted from the UI.
        """
        collection_name = self.get_thread_collection_name(thread_id)
        
        if not self.client.collection_exists(collection_name):
            return False

        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="message_id",
                                match=models.MatchValue(value=message_id)
                            )
                        ]
                    )
                )
            )
            print(f"Deleted chat memory for message {message_id}", flush=True)
            return True
        except Exception as e:
            print(f"Error deleting chat memory: {e}", flush=True)
            return False

    # ============ Legacy Operations (Backward Compatibility) ============

    async def index_documents(
        self,
        collection_name: str,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> None:
        """
        Legacy method: Index documents into the specified Qdrant collection.
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
        Legacy method: Search for similar vectors in the specified collection.
        Returns a list of dicts with text, metadata, and score.
        """
        if not self.client.collection_exists(collection_name):
            return []
            
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

    # ============ Utility Methods ============

    async def get_collection_stats(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a collection."""
        try:
            if not self.client.collection_exists(collection_name):
                return None
            
            info = self.client.get_collection(collection_name)
            # Handle different qdrant-client versions - vectors_count may not exist
            vectors_count = getattr(info, 'vectors_count', None)
            if vectors_count is None:
                # Try to get from indexed_vectors_count or just use points_count
                vectors_count = getattr(info, 'indexed_vectors_count', info.points_count)
            
            return {
                "points_count": info.points_count,
                "vectors_count": vectors_count,
                "status": info.status
            }
        except Exception as e:
            print(f"Error getting collection stats: {e}", flush=True)
            return None

    async def get_thread_stats(self, thread_id: str) -> Dict[str, Any]:
        """Get statistics for a thread's collection."""
        collection_name = self.get_thread_collection_name(thread_id)
        stats = await self.get_collection_stats(collection_name)
        
        if not stats:
            return {"exists": False, "pdf_chunks": 0, "chat_memories": 0}
        
        # Count by type
        try:
            pdf_count = self.client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="type", match=models.MatchValue(value="pdf_chunk"))]
                )
            ).count
            
            chat_count = self.client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="type", match=models.MatchValue(value="chat_memory"))]
                )
            ).count
            
            return {
                "exists": True,
                "pdf_chunks": pdf_count,
                "chat_memories": chat_count,
                "total_points": stats["points_count"]
            }
        except Exception as e:
            print(f"Error counting points: {e}", flush=True)
            return {"exists": True, **stats}

    async def has_file_indexed(self, thread_id: str, file_hash: str) -> bool:
        """Check if a specific file has been indexed in a thread's collection."""
        collection_name = self.get_thread_collection_name(thread_id)
        if not await self.thread_collection_exists(thread_id):
            return False
        
        try:
            count = self.client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="type", match=models.MatchValue(value="pdf_chunk")),
                        models.FieldCondition(key="file_hash", match=models.MatchValue(value=file_hash))
                    ]
                )
            ).count
            return count > 0
        except Exception as e:
            print(f"Error checking file indexing: {e}", flush=True)
            return False
