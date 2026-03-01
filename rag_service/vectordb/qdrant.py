"""
qdrant.py - Qdrant vector database adapter with per-thread collections

This adapter supports:
- Per-thread collections (col_thread_{thread_id})
- Knowledge source storage (PDFs and webpages) with file_hash filtering
- Chat QA memory storage and retrieval
- Semantic search across knowledge sources and chat content

Data types stored in each thread collection:
- knowledge_source  (source_kind: 'pdf' | 'webpage')  — uploaded documents & indexed webpages
- chat_memory       — past QA conversation pairs
- web_search        — cached internet search result snippets
"""

from typing import List, Dict, Any, Optional
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models


# Module-level singleton — avoids re-creating a TCP connection on every call.
_singleton_instance: Optional["QdrantAdapter"] = None


def get_qdrant() -> "QdrantAdapter":
    """Return the shared QdrantAdapter singleton."""
    global _singleton_instance
    if _singleton_instance is None:
        _singleton_instance = QdrantAdapter()
    return _singleton_instance


class QdrantAdapter:
    """
    Adapter for Qdrant vector database.
    Supports per-thread collections and dual-search (knowledge sources + chat memory).

    Prefer using the module-level ``get_qdrant()`` factory to obtain a shared
    singleton instance rather than instantiating this class directly.
    """
    def __init__(self) -> None:
        """Initialize the Qdrant client using environment variables for host and port."""
        host = os.getenv("QDRANT_HOST")
        port_str = os.getenv("QDRANT_PORT")
        if host is None:
            raise ValueError("QDRANT_HOST environment variable is not set")
        if port_str is None:
            raise ValueError("QDRANT_PORT environment variable is not set")
        self.client = QdrantClient(host=host, port=int(port_str))

    # ============ Collection Naming Helpers ============
    
    @staticmethod
    def get_thread_collection_name(thread_id: str) -> str:
        """Get the collection name for a thread."""
        return f"col_thread_{thread_id}"
    
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
            # Create payload indexes for commonly filtered fields to speed up
            # filtered vector searches as the collection grows.
            for field, schema in [
                ("type",        models.PayloadSchemaType.KEYWORD),
                ("file_hash",   models.PayloadSchemaType.KEYWORD),
                ("message_id",  models.PayloadSchemaType.KEYWORD),
                ("source_kind", models.PayloadSchemaType.KEYWORD),
                ("chunk_id",    models.PayloadSchemaType.INTEGER),
            ]:
                try:
                    self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=schema,
                    )
                except Exception as idx_err:
                    print(f"Warning: could not create index on '{field}': {idx_err}", flush=True)
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

    # ============ Knowledge Source Operations (PDF + Webpage) ============

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
                "type": "knowledge_source",
                "source_kind": "pdf",
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

    async def search_knowledge_sources(
        self,
        thread_id: str,
        query_vector: List[float],
        limit: int = 5,
        file_hash: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for indexed knowledge source chunks (PDFs and webpages) in a thread's collection.
        Optionally filter by file_hash.
        """
        collection_name = self.get_thread_collection_name(thread_id)
        
        if not self.client.collection_exists(collection_name):
            return []

        # Build filter for knowledge sources (PDFs and webpages share type='knowledge_source')
        must_conditions = [
            models.FieldCondition(
                key="type",
                match=models.MatchValue(value="knowledge_source")
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
            source_kind = hit.payload.get("source_kind", "pdf")
            results.append({
                "text": hit.payload.get("text", ""),
                "file_hash": hit.payload.get("file_hash"),
                "chunk_id": hit.payload.get("chunk_id"),
                "type": "knowledge_source",
                "source_kind": source_kind,
                "url": hit.payload.get("url"),
                "title": hit.payload.get("title"),
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
            models.FieldCondition(key="type", match=models.MatchValue(value="knowledge_source")),
            models.FieldCondition(key="source_kind", match=models.MatchValue(value="pdf")),
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
                "type": "knowledge_source",
                "source_kind": "pdf",
                "metadata": {k: v for k, v in hit.payload.items() if k not in ["text", "type", "source_kind"]}
            })
        
        # Sort by chunk_id to maintain sequence
        results.sort(key=lambda x: x["chunk_id"])
        return results

    # ============ Chat Memory Operations ============

    async def index_chat_memory(
        self,
        thread_id: str,
        message_id: str,
        question: str,
        answer: str,
        texts: List[str],
        embeddings: List[List[float]]
    ) -> int:
        """
        Store a QA pair as chat memory for semantic retrieval.
        Handles multiple chunks to preserve more context.
        """
        collection_name = self.get_thread_collection_name(thread_id)
        
        if not self.client.collection_exists(collection_name):
            await self.create_thread_collection(thread_id, len(embeddings[0]))

        points = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            payload = {
                "text": text,
                "type": "chat_memory",
                "message_id": message_id,
                "thread_id": thread_id,
                "question": question,
                "answer": answer,
                "chunk_id": i
            }
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        print(f"Stored {len(points)} chat memory chunks for message {message_id} in thread {thread_id}", flush=True)
        return len(points)

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

    # ============ Web Search Chunk Operations ============

    async def index_web_search_chunks(
        self,
        thread_id: str,
        query: str,
        texts: List[str],
        embeddings: List[List[float]],
        urls: Optional[List[str]] = None,
        titles: Optional[List[str]] = None,
    ) -> int:
        """
        Store web search result snippets as indexed chunks for future retrieval.
        Each snippet is stored with type='web_search' alongside its source URL/title.
        Returns the number of chunks indexed.
        """
        if not embeddings:
            return 0

        collection_name = self.get_thread_collection_name(thread_id)
        if not self.client.collection_exists(collection_name):
            await self.create_thread_collection(thread_id, len(embeddings[0]))

        points = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            payload = {
                "text": text,
                "type": "web_search",
                "search_query": query,
                "url": (urls[i] if urls and i < len(urls) else ""),
                "title": (titles[i] if titles and i < len(titles) else ""),
                "chunk_id": i,
            }
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Indexed {len(points)} web search chunks for thread {thread_id}", flush=True)
        return len(points)

    async def search_web_chunks(
        self,
        thread_id: str,
        query_vector: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over previously indexed web search results in a thread's collection.
        Returns ranked results with URL, title, and snippet text.
        """
        collection_name = self.get_thread_collection_name(thread_id)
        if not self.client.collection_exists(collection_name):
            return []

        search_filter = models.Filter(
            must=[
                models.FieldCondition(key="type", match=models.MatchValue(value="web_search"))
            ]
        )

        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
        ).points

        results = []
        for hit in search_result:
            results.append({
                "text": hit.payload.get("text", ""),
                "url": hit.payload.get("url", ""),
                "title": hit.payload.get("title", ""),
                "search_query": hit.payload.get("search_query", ""),
                "type": "web_search",
                "score": hit.score,
            })
        return results

    async def delete_web_chunks_by_urls(
        self,
        thread_id: str,
        urls: List[str],
    ) -> int:
        """
        Delete all web_search points whose URL matches any of the given URLs.
        Returns the number of URLs targeted for deletion.
        Used when a QA message pair is deleted to clean up orphaned web search chunks.
        """
        if not urls:
            return 0
        collection_name = self.get_thread_collection_name(thread_id)
        if not self.client.collection_exists(collection_name):
            return 0
        try:
            self.client.delete(
                collection_name=collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="type",
                                match=models.MatchValue(value="web_search"),
                            ),
                            models.FieldCondition(
                                key="url",
                                match=models.MatchAny(any=urls),
                            ),
                        ]
                    )
                ),
            )
            print(f"Deleted web_search chunks for {len(urls)} URL(s) in thread {thread_id}", flush=True)
            return len(urls)
        except Exception as e:
            print(f"Error deleting web_search chunks by URL: {e}", flush=True)
            return 0

    # ============ Web Source (Indexed Webpage) Operations ============

    async def index_web_source_chunks(
        self,
        thread_id: str,
        file_hash: str,
        url: str,
        title: str,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Index webpage content chunks into a thread's collection.
        Stored with type='knowledge_source', source_kind='webpage' and file_hash for later deletion.
        Returns the number of chunks indexed.
        """
        if not embeddings:
            return 0

        collection_name = self.get_thread_collection_name(thread_id)
        if not self.client.collection_exists(collection_name):
            await self.create_thread_collection(thread_id, len(embeddings[0]))

        points = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            metadata = metadatas[i] if metadatas else {}
            payload = {
                "text": text,
                "type": "knowledge_source",
                "source_kind": "webpage",
                "file_hash": file_hash,
                "url": url,
                "title": title,
                "chunk_id": i,
                **metadata,
            }
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Indexed {len(points)} web source chunks for thread {thread_id}, url {url}", flush=True)
        return len(points)

    async def delete_source_chunks_by_file_hash(
        self,
        thread_id: str,
        file_hash: str,
    ) -> bool:
        """
        Delete all knowledge_source chunks belonging to a given file_hash
        from a thread's collection.  Used when a source is removed from a thread.
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
                                key="file_hash",
                                match=models.MatchValue(value=file_hash),
                            )
                        ]
                    )
                ),
            )
            print(f"Deleted chunks for file_hash {file_hash} in thread {thread_id}", flush=True)
            return True
        except Exception as e:
            print(f"Error deleting source chunks: {e}", flush=True)
            return False

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
            return {"exists": False, "knowledge_sources": 0, "chat_memories": 0}
        
        # Count by type
        try:
            knowledge_count = self.client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="type", match=models.MatchValue(value="knowledge_source"))]
                )
            ).count

            pdf_count = self.client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="type", match=models.MatchValue(value="knowledge_source")),
                        models.FieldCondition(key="source_kind", match=models.MatchValue(value="pdf")),
                    ]
                )
            ).count

            webpage_count = self.client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(key="type", match=models.MatchValue(value="knowledge_source")),
                        models.FieldCondition(key="source_kind", match=models.MatchValue(value="webpage")),
                    ]
                )
            ).count
            
            chat_count = self.client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="type", match=models.MatchValue(value="chat_memory"))]
                )
            ).count

            web_search_count = self.client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[models.FieldCondition(key="type", match=models.MatchValue(value="web_search"))]
                )
            ).count
            
            return {
                "exists": True,
                "knowledge_sources": knowledge_count,
                "pdf_chunks": pdf_count,
                "webpage_chunks": webpage_count,
                "chat_memories": chat_count,
                "web_search_chunks": web_search_count,
                "total_points": stats["points_count"]
            }
        except Exception as e:
            print(f"Error counting points: {e}", flush=True)
            return {"exists": True, **stats}

    async def has_file_indexed(self, thread_id: str, file_hash: str) -> bool:
        """Check if a specific file has been indexed in a thread's collection (any knowledge source)."""
        collection_name = self.get_thread_collection_name(thread_id)
        if not await self.thread_collection_exists(thread_id):
            return False
        
        try:
            count = self.client.count(
                collection_name=collection_name,
                count_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="type",
                            match=models.MatchValue(value="knowledge_source")
                        ),
                        models.FieldCondition(key="file_hash", match=models.MatchValue(value=file_hash))
                    ]
                )
            ).count
            return count > 0
        except Exception as e:
            print(f"Error checking file indexing: {e}", flush=True)
            return False
