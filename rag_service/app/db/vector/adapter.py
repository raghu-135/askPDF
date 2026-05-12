"""
vector/adapter.py - Weaviate adapter with thread-filtered global collections.

Collections:
- DocumentChunk   : PDF and indexed webpage chunks
- ChatMemoryChunk : semantic memory chunks from QA pairs
- WebSearchChunk  : cached internet search snippets
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import weaviate
import weaviate.classes as wvc
from weaviate.exceptions import WeaviateBaseError, WeaviateConnectionError

from app.db.vector.config import (
    CollectionNames,
    VectorDBError,
    VectorDBConnectionError,
    VectorDBInsertError,
    VectorDBQueryError,
)
from app.db.vector.helpers import (
    _validate_not_empty,
    _validate_embeddings_match_texts,
    _metadata_json,
    _parse_metadata,
    _score,
)
from app.db.vector.model_registry import get_embedding_model_registry
from app.db.vector.collection_manager import ModelAwareCollectionManager

logger = logging.getLogger(__name__)

_singleton_instance: Optional["WeaviateAdapter"] = None


class WeaviateAdapter:
    """Adapter for Weaviate with thread-filtered global collections.
    
    This adapter provides async wrappers around Weaviate SDK operations,
    with application-specific collection management and query logic.
    """

    def __init__(self) -> None:
        """Initialize Weaviate client connection and ensure required collections exist.
        
        Raises:
            ValueError: If WEAVIATE_URL environment variable is not set.
            VectorDBConnectionError: If connection to Weaviate fails.
        """
        weaviate_url = os.getenv("WEAVIATE_URL")
        if not weaviate_url:
            raise ValueError("WEAVIATE_URL environment variable is not set")

        _hybrid_alpha = os.environ.get("WEAVIATE_HYBRID_ALPHA")
        if _hybrid_alpha is None:
            raise RuntimeError("WEAVIATE_HYBRID_ALPHA environment variable is required")
        self.hybrid_alpha = float(_hybrid_alpha)
        parsed = urlparse(weaviate_url if "://" in weaviate_url else f"http://{weaviate_url}")
        host = parsed.hostname or "weaviate"
        port = parsed.port or 8080
        secure = parsed.scheme == "https"

        logger.info(f"Connecting to Weaviate at {host}:{port} (secure={secure})")
        
        try:
            self.client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=secure,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=secure,
                skip_init_checks=True,
            )
            logger.info("Successfully connected to Weaviate")
        except WeaviateConnectionError as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise VectorDBConnectionError(f"Could not connect to Weaviate at {host}:{port}") from e

        self.collection_manager = ModelAwareCollectionManager(self.client)
        # Keep legacy collections for backward compatibility during migration
        self._ensure_collections_sync()

    def __enter__(self) -> "WeaviateAdapter":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - close connection."""
        self.close()

    def close(self) -> None:
        """Close the Weaviate client connection."""
        if self.client:
            try:
                self.client.close()
                logger.info("Weaviate client connection closed")
            except Exception as e:
                logger.warning(f"Error closing Weaviate client: {e}")

    async def health_check(self) -> bool:
        """Check if the Weaviate connection is healthy.
        
        Returns:
            bool: True if connection is healthy, False otherwise.
        """
        try:
            is_ready = await asyncio.to_thread(self.client.is_ready)
            if is_ready:
                logger.debug("Weaviate health check passed")
            else:
                logger.warning("Weaviate health check failed - not ready")
            return is_ready
        except Exception as e:
            logger.error(f"Weaviate health check error: {e}")
            return False

    async def ensure_collections(self) -> None:
        """Ensure all required Weaviate collections exist (async wrapper)."""
        await asyncio.to_thread(self._ensure_collections_sync)

    def _ensure_collections_sync(self) -> None:
        """Synchronously create all application collections if missing."""
        logger.info("Ensuring Weaviate collections exist")
        self._ensure_collection(
            CollectionNames.DOCUMENT,
            [
                ("thread_id", wvc.config.DataType.TEXT),
                ("type", wvc.config.DataType.TEXT),
                ("embed_model", wvc.config.DataType.TEXT),
                ("source_kind", wvc.config.DataType.TEXT),
                ("file_hash", wvc.config.DataType.TEXT),
                ("chunk_id", wvc.config.DataType.INT),
                ("text", wvc.config.DataType.TEXT),
                ("url", wvc.config.DataType.TEXT),
                ("title", wvc.config.DataType.TEXT),
                ("metadata_json", wvc.config.DataType.TEXT),
            ],
        )
        self._ensure_collection(
            CollectionNames.CHAT_MEMORY,
            [
                ("thread_id", wvc.config.DataType.TEXT),
                ("type", wvc.config.DataType.TEXT),
                ("message_id", wvc.config.DataType.TEXT),
                ("chunk_id", wvc.config.DataType.INT),
                ("question", wvc.config.DataType.TEXT),
                ("answer", wvc.config.DataType.TEXT),
                ("text", wvc.config.DataType.TEXT),
            ],
        )
        self._ensure_collection(
            CollectionNames.WEB_SEARCH,
            [
                ("thread_id", wvc.config.DataType.TEXT),
                ("type", wvc.config.DataType.TEXT),
                ("search_query", wvc.config.DataType.TEXT),
                ("chunk_id", wvc.config.DataType.INT),
                ("text", wvc.config.DataType.TEXT),
                ("url", wvc.config.DataType.TEXT),
                ("title", wvc.config.DataType.TEXT),
            ],
        )
        logger.info("All Weaviate collections ensured")

    def _ensure_collection(self, name: str, properties: List[tuple[str, Any]]) -> None:
        """Create a single collection with self-provided vectors when absent.
        
        Args:
            name: Collection name.
            properties: List of (property_name, data_type) tuples.
        """
        if self.client.collections.exists(name):
            logger.debug(f"Collection '{name}' already exists")
            return
        
        logger.warning(f"Collection '{name}' is missing, creating it now...")
        try:
            logger.info(f"Creating collection '{name}' with {len(properties)} properties for vector storage")
            self.client.collections.create(
                name=name,
                vector_config=wvc.config.Configure.Vectors.self_provided(),
                properties=[
                    wvc.config.Property(name=prop_name, data_type=prop_dtype)
                    for prop_name, prop_dtype in properties
                ],
            )
            logger.info(f"Successfully created collection '{name}' - ready for vector embeddings")
        except WeaviateBaseError as e:
            logger.error(f"Failed to create collection '{name}': {e}")
            raise VectorDBError(f"Could not create collection '{name}'") from e

    async def _insert_many_model_aware(self, collection, points: List[Dict[str, Any]]) -> int:
        """Insert vector points into a model-aware collection and return inserted count.
        
        Args:
            collection: Weaviate collection object (already model-aware).
            points: List of points with 'properties' and 'vector' keys.
            
        Returns:
            int: Number of points inserted.
            
        Raises:
            VectorDBInsertError: If insertion fails.
        """
        if not points:
            logger.debug("No points to insert into collection")
            return 0
        
        inserted_count = 0
        
        try:
            # Use batch insert for better performance
            with collection.batch.dynamic() as batch:
                for p in points:
                    batch.add_object(
                        properties=p["properties"],
                        vector=p["vector"],
                        uuid=str(uuid.uuid4()),
                    )
                    inserted_count += 1
            
            logger.info(f"Inserted {inserted_count} points into model-aware collection")
            return inserted_count
        except WeaviateBaseError as e:
            logger.error(f"Failed to insert points into model-aware collection: {e}")
            raise VectorDBInsertError("Could not insert points into model-aware collection") from e
        except Exception as e:
            logger.error(f"Unexpected error during insert into model-aware collection: {e}")
            raise VectorDBInsertError("Unexpected error inserting into model-aware collection") from e

    async def _insert_many(self, collection_name: str, points: List[Dict[str, Any]]) -> int:
        """Insert vector points into a target collection and return inserted count.
        
        Args:
            collection_name: Name of the collection to insert into.
            points: List of points with 'properties' and 'vector' keys.
            
        Returns:
            int: Number of points inserted.
            
        Raises:
            VectorDBInsertError: If insertion fails.
        """
        if not points:
            logger.debug(f"No points to insert into collection '{collection_name}'")
            return 0
        
        col = self.client.collections.use(collection_name)
        inserted_count = 0
        
        try:
            # Use batch insert for better performance
            with col.batch.dynamic() as batch:
                for p in points:
                    batch.add_object(
                        properties=p["properties"],
                        vector=p["vector"],
                        uuid=str(uuid.uuid4()),
                    )
                    inserted_count += 1
            
            logger.info(f"Inserted {inserted_count} points into collection '{collection_name}'")
            return inserted_count
        except WeaviateBaseError as e:
            logger.error(f"Failed to insert points into collection '{collection_name}': {e}")
            raise VectorDBInsertError(f"Could not insert points into '{collection_name}'") from e
        except Exception as e:
            logger.error(f"Unexpected error during insert into '{collection_name}': {e}")
            raise VectorDBInsertError(f"Unexpected error inserting into '{collection_name}'") from e

    async def delete_thread_data(self, thread_id: str) -> bool:
        """Delete only thread-scoped chat-memory and web-search vectors for a thread.
        
        Args:
            thread_id: Thread identifier.
            
        Returns:
            bool: True if deletion succeeded, False otherwise.
        """
        _validate_not_empty(thread_id, "thread_id")
        filt = wvc.query.Filter.by_property("thread_id").equal(thread_id)
        
        try:
            for name in [CollectionNames.CHAT_MEMORY, CollectionNames.WEB_SEARCH]:
                col = self.client.collections.use(name)
                await asyncio.to_thread(col.data.delete_many, where=filt)
            logger.info(f"Deleted thread data for thread '{thread_id}'")
            return True
        except WeaviateBaseError as e:
            logger.error(f"Failed to delete thread data for '{thread_id}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting thread data for '{thread_id}': {e}")
            return False

    async def index_pdf_chunks(
        self,
        thread_id: str,
        embedding_model_name: str,
        file_hash: str,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """Index PDF chunks into DocumentChunk with embedding-model/file identifiers.
        
        Args:
            thread_id: Thread identifier.
            embedding_model_name: Name of the embedding model used.
            file_hash: Hash of the source file.
            texts: List of text chunks to index.
            embeddings: List of embedding vectors corresponding to texts.
            metadatas: Optional list of metadata dictionaries for each chunk.
            
        Returns:
            int: Number of chunks indexed.
            
        Raises:
            ValueError: If validation fails.
            VectorDBInsertError: If insertion fails.
        """
        _validate_not_empty(thread_id, "thread_id")
        _validate_not_empty(embedding_model_name, "embedding_model_name")
        _validate_not_empty(file_hash, "file_hash")
        _validate_embeddings_match_texts(texts, embeddings)
        
        logger.info(f"Indexing {len(texts)} PDF chunks for thread '{thread_id}', file '{file_hash}'")
        
        points: List[Dict[str, Any]] = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            md = metadatas[i] if metadatas and i < len(metadatas) else {}
            source_kind = md.get("source_kind", "pdf")
            url = md.get("url") or md.get("original_url") or ""
            title = md.get("title") or ""
            points.append(
                {
                    "vector": vector,
                    "properties": {
                        "thread_id": thread_id,
                        "type": "knowledge_source",
                        "embed_model": embedding_model_name,
                        "source_kind": source_kind,
                        "file_hash": file_hash,
                        "chunk_id": i,
                        "text": text,
                        "url": url,
                        "title": title,
                        "metadata_json": _metadata_json(md),
                    },
                }
            )
        # Use model-aware collection manager
        collection = await self.collection_manager.get_collection(CollectionNames.DOCUMENT, embedding_model_name)
        return await self._insert_many_model_aware(collection, points)

    async def index_chat_memory(
        self,
        thread_id: str,
        message_id: str,
        question: str,
        answer: str,
        texts: List[str],
        embeddings: List[List[float]],
        embedding_model_name: str,
    ) -> int:
        """Index compact chat-memory chunks into ChatMemoryChunk.
        
        Args:
            thread_id: Thread identifier.
            message_id: Message identifier.
            question: User question.
            answer: Assistant answer.
            texts: List of text chunks to index.
            embeddings: List of embedding vectors corresponding to texts.
            embedding_model_name: Name of the embedding model used.
            
        Returns:
            int: Number of chunks indexed.
            
        Raises:
            ValueError: If validation fails.
            VectorDBInsertError: If insertion fails.
        """
        _validate_not_empty(thread_id, "thread_id")
        _validate_not_empty(message_id, "message_id")
        _validate_not_empty(question, "question")
        _validate_not_empty(answer, "answer")
        _validate_not_empty(embedding_model_name, "embedding_model_name")
        _validate_embeddings_match_texts(texts, embeddings)
        
        logger.info(f"Indexing {len(texts)} chat memory chunks for thread '{thread_id}', message '{message_id}'")
        
        points: List[Dict[str, Any]] = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            points.append(
                {
                    "vector": vector,
                    "properties": {
                        "thread_id": thread_id,
                        "type": "chat_memory",
                        "message_id": message_id,
                        "chunk_id": i,
                        "question": question,
                        "answer": answer,
                        "text": text,
                    },
                }
            )
        # Use model-aware collection manager
        collection = await self.collection_manager.get_collection(CollectionNames.CHAT_MEMORY, embedding_model_name)
        return await self._insert_many_model_aware(collection, points)

    async def index_web_search_chunks(
        self,
        thread_id: str,
        query: str,
        texts: List[str],
        embeddings: List[List[float]],
        embedding_model_name: str,
        urls: Optional[List[str]] = None,
        titles: Optional[List[str]] = None,
    ) -> int:
        """Index transient web-search snippets into WebSearchChunk.
        
        Args:
            thread_id: Thread identifier.
            query: Search query that generated these results.
            texts: List of text chunks to index.
            embeddings: List of embedding vectors corresponding to texts.
            embedding_model_name: Name of the embedding model used.
            urls: Optional list of source URLs.
            titles: Optional list of page titles.
            
        Returns:
            int: Number of chunks indexed.
            
        Raises:
            ValueError: If validation fails.
            VectorDBInsertError: If insertion fails.
        """
        _validate_not_empty(thread_id, "thread_id")
        _validate_not_empty(query, "query")
        _validate_not_empty(embedding_model_name, "embedding_model_name")
        _validate_embeddings_match_texts(texts, embeddings)
        
        logger.info(f"Indexing {len(texts)} web search chunks for thread '{thread_id}', query '{query}'")
        
        points: List[Dict[str, Any]] = []
        for i, (text, vector) in enumerate(zip(texts, embeddings)):
            points.append(
                {
                    "vector": vector,
                    "properties": {
                        "thread_id": thread_id,
                        "type": "web_search",
                        "search_query": query,
                        "chunk_id": i,
                        "text": text,
                        "url": urls[i] if urls and i < len(urls) else "",
                        "title": titles[i] if titles and i < len(titles) else "",
                    },
                }
            )
        # Use model-aware collection manager
        collection = await self.collection_manager.get_collection(CollectionNames.WEB_SEARCH, embedding_model_name)
        return await self._insert_many_model_aware(collection, points)

    async def search_knowledge_sources(
        self,
        thread_id: str,
        query_vector: List[float],
        embedding_model_name: str,
        limit: int = 5,
        file_hash: Optional[str] = None,
        file_hashes: Optional[List[str]] = None,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search document chunks using vector or hybrid retrieval with file/model filters.
        
        Args:
            thread_id: Thread identifier (for logging/context).
            query_vector: Query embedding vector.
            embedding_model_name: Name of embedding model to filter by.
            limit: Maximum number of results to return.
            file_hash: Optional specific file hash to filter by.
            file_hashes: Optional list of file hashes to filter by.
            query_text: Optional query text for hybrid search.
            
        Returns:
            List[Dict[str, Any]]: List of search results with text, metadata, and scores.
            
        Raises:
            ValueError: If validation fails.
            VectorDBQueryError: If query fails.
        """
        _validate_not_empty(thread_id, "thread_id")
        _validate_not_empty(query_vector, "query_vector")
        _validate_not_empty(embedding_model_name, "embedding_model_name")
        if limit <= 0:
            raise ValueError("limit must be positive")
        
        logger.debug(f"Searching knowledge sources for thread '{thread_id}', model '{embedding_model_name}', limit={limit}")
        
        # Use model-aware collection - no need for embed_model filter
        col = await self.collection_manager.get_collection(CollectionNames.DOCUMENT, embedding_model_name)
        
        # Only thread and file filters needed
        base_filter = wvc.query.Filter.by_property("thread_id").equal(thread_id)
        if file_hash:
            base_filter = base_filter & wvc.query.Filter.by_property("file_hash").equal(file_hash)
        if file_hashes:
            base_filter = base_filter & wvc.query.Filter.by_property("file_hash").contains_any(file_hashes)
        kwargs = {
            "filters": base_filter,
            "limit": limit,
            "return_metadata": wvc.query.MetadataQuery(score=True),
        }

        try:
            if query_text:
                response = await asyncio.to_thread(
                    col.query.hybrid,
                    query=query_text,
                    vector=query_vector,
                    alpha=self.hybrid_alpha,
                    **kwargs,
                )
            else:
                response = await asyncio.to_thread(
                    col.query.near_vector,
                    near_vector=query_vector,
                    **kwargs,
                )
        except WeaviateBaseError as e:
            logger.error(f"Failed to search knowledge sources: {e}")
            raise VectorDBQueryError("Could not search knowledge sources") from e

        results: List[Dict[str, Any]] = []
        for obj in response.objects:
            p = obj.properties
            results.append(
                {
                    "text": p.get("text", ""),
                    "file_hash": p.get("file_hash"),
                    "chunk_id": p.get("chunk_id"),
                    "type": p.get("type", "knowledge_source"),
                    "source_kind": p.get("source_kind", "pdf"),
                    "url": p.get("url"),
                    "title": p.get("title"),
                    "score": _score(obj),
                    "metadata": _parse_metadata(p.get("metadata_json")),
                }
            )
        
        logger.debug(f"Found {len(results)} knowledge source results")
        return results

    async def get_knowledge_source_chunks_by_ids(
        self,
        thread_id: str,
        embedding_model_name: str,
        file_hash: str,
        chunk_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """Fetch specific document chunks by file hash and chunk IDs.
        
        Args:
            thread_id: Thread identifier (for logging/context).
            embedding_model_name: Name of embedding model.
            file_hash: File hash to filter by.
            chunk_ids: List of chunk IDs to fetch.
            
        Returns:
            List[Dict[str, Any]]: List of chunks sorted by chunk_id.
            
        Raises:
            ValueError: If validation fails.
            VectorDBQueryError: If query fails.
        """
        _validate_not_empty(thread_id, "thread_id")
        _validate_not_empty(embedding_model_name, "embedding_model_name")
        _validate_not_empty(file_hash, "file_hash")
        if not chunk_ids:
            return []
        
        logger.debug(f"Fetching {len(chunk_ids)} chunks for file '{file_hash}'")
        
        # Use model-aware collection - no need for embed_model filter
        col = await self.collection_manager.get_collection(CollectionNames.DOCUMENT, embedding_model_name)
        
        filt = (
            wvc.query.Filter.by_property("file_hash").equal(file_hash)
            & wvc.query.Filter.by_property("chunk_id").contains_any(chunk_ids)
        )
        try:
            response = await asyncio.to_thread(
                col.query.fetch_objects,
                filters=filt,
                limit=len(chunk_ids),
                return_metadata=wvc.query.MetadataQuery(score=True),
            )
        except WeaviateBaseError as e:
            logger.error(f"Failed to fetch chunks by IDs: {e}")
            raise VectorDBQueryError("Could not fetch chunks by IDs") from e

        out: List[Dict[str, Any]] = []
        for obj in response.objects:
            p = obj.properties
            out.append(
                {
                    "text": p.get("text", ""),
                    "file_hash": p.get("file_hash"),
                    "chunk_id": p.get("chunk_id"),
                    "type": p.get("type", "knowledge_source"),
                    "source_kind": p.get("source_kind", "pdf"),
                    "url": p.get("url"),
                    "title": p.get("title"),
                    "metadata": _parse_metadata(p.get("metadata_json")),
                }
            )

        out.sort(key=lambda x: x.get("chunk_id", 0))
        logger.debug(f"Retrieved {len(out)} chunks")
        return out

    async def search_chat_memory(
        self,
        thread_id: str,
        query_vector: List[float],
        limit: int = 3,
        exclude_message_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search chat-memory vectors within a thread, with optional message exclusions.
        
        Args:
            thread_id: Thread identifier.
            query_vector: Query embedding vector.
            limit: Maximum number of results to return.
            exclude_message_ids: Optional list of message IDs to exclude.
            
        Returns:
            List[Dict[str, Any]]: List of chat memory results.
            
        Raises:
            ValueError: If validation fails.
            VectorDBQueryError: If query fails.
        """
        _validate_not_empty(thread_id, "thread_id")
        _validate_not_empty(query_vector, "query_vector")
        if limit <= 0:
            raise ValueError("limit must be positive")
        
        logger.debug(f"Searching chat memory for thread '{thread_id}', limit={limit}")
        
        # Use model-aware collection - no need for embed_model filter
        col = await self.collection_manager.get_collection(CollectionNames.CHAT_MEMORY, embedding_model_name)
        
        filt = wvc.query.Filter.by_property("thread_id").equal(thread_id)
        if exclude_message_ids:
            filt = filt & wvc.query.Filter.not_(
                wvc.query.Filter.by_property("message_id").contains_any(exclude_message_ids)
            )
        try:
            response = await asyncio.to_thread(
                col.query.near_vector,
                near_vector=query_vector,
                filters=filt,
                limit=limit,
                return_metadata=wvc.query.MetadataQuery(score=True),
            )
        except WeaviateBaseError as e:
            logger.error(f"Failed to search chat memory: {e}")
            raise VectorDBQueryError("Could not search chat memory") from e

        out: List[Dict[str, Any]] = []
        for obj in response.objects:
            p = obj.properties
            out.append(
                {
                    "text": p.get("text", ""),
                    "message_id": p.get("message_id"),
                    "question": p.get("question"),
                    "answer": p.get("answer"),
                    "type": p.get("type", "chat_memory"),
                    "score": _score(obj),
                }
            )
        
        logger.debug(f"Found {len(out)} chat memory results")
        return out

    async def search_web_chunks(
        self,
        thread_id: str,
        query_vector: List[float],
        limit: int = 5,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search web-search snippet vectors for a thread via vector or hybrid query.
        
        Args:
            thread_id: Thread identifier.
            query_vector: Query embedding vector.
            limit: Maximum number of results to return.
            query_text: Optional query text for hybrid search.
            
        Returns:
            List[Dict[str, Any]]: List of web search results.
            
        Raises:
            ValueError: If validation fails.
            VectorDBQueryError: If query fails.
        """
        _validate_not_empty(thread_id, "thread_id")
        _validate_not_empty(query_vector, "query_vector")
        if limit <= 0:
            raise ValueError("limit must be positive")
        
        logger.debug(f"Searching web chunks for thread '{thread_id}', limit={limit}")
        
        # Use model-aware collection - no need for embed_model filter
        col = await self.collection_manager.get_collection(CollectionNames.WEB_SEARCH, embedding_model_name)
        
        filt = wvc.query.Filter.by_property("thread_id").equal(thread_id)

        kwargs = {
            "filters": filt,
            "limit": limit,
            "return_metadata": wvc.query.MetadataQuery(score=True),
        }

        try:
            if query_text:
                response = await asyncio.to_thread(
                    col.query.hybrid,
                    query=query_text,
                    vector=query_vector,
                    alpha=self.hybrid_alpha,
                    **kwargs,
                )
            else:
                response = await asyncio.to_thread(
                    col.query.near_vector,
                    near_vector=query_vector,
                    **kwargs,
                )
        except WeaviateBaseError as e:
            logger.error(f"Failed to search web chunks: {e}")
            raise VectorDBQueryError("Could not search web chunks") from e

        out: List[Dict[str, Any]] = []
        for obj in response.objects:
            p = obj.properties
            out.append(
                {
                    "text": p.get("text", ""),
                    "url": p.get("url", ""),
                    "title": p.get("title", ""),
                    "search_query": p.get("search_query", ""),
                    "type": p.get("type", "web_search"),
                    "score": _score(obj),
                }
            )
        
        logger.debug(f"Found {len(out)} web chunk results")
        return out

    async def delete_chat_memory_by_message_id(self, thread_id: str, message_id: str, embedding_model_name: str) -> bool:
        """Delete all chat-memory chunks belonging to a single assistant message.
        
        Args:
            thread_id: Thread identifier.
            message_id: Message identifier.
            embedding_model_name: Name of the embedding model used for the chat memory.
            
        Returns:
            bool: True if deletion succeeded, False otherwise.
        """
        _validate_not_empty(thread_id, "thread_id")
        _validate_not_empty(message_id, "message_id")
        _validate_not_empty(embedding_model_name, "embedding_model_name")
        
        # Use model-aware collection
        col = await self.collection_manager.get_collection(CollectionNames.CHAT_MEMORY, embedding_model_name)
        
        filt = (
            wvc.query.Filter.by_property("thread_id").equal(thread_id)
            & wvc.query.Filter.by_property("message_id").equal(message_id)
        )
        
        try:
            await asyncio.to_thread(col.data.delete_many, where=filt)
            logger.info(f"Deleted chat memory for message '{message_id}' in thread '{thread_id}'")
            return True
        except WeaviateBaseError as e:
            logger.error(f"Failed to delete chat memory for message '{message_id}': {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting chat memory for message '{message_id}': {e}")
            return False

    async def delete_web_chunks_by_urls(self, thread_id: str, urls: List[str], embedding_model_name: str) -> int:
        """Delete web-search chunks for a thread whose URLs match any provided URL.
        
        Args:
            thread_id: Thread identifier.
            urls: List of URLs to delete.
            embedding_model_name: Name of embedding model used for web search chunks.
            
        Returns:
            int: Number of URLs processed (not necessarily deleted count).
        """
        _validate_not_empty(thread_id, "thread_id")
        _validate_not_empty(embedding_model_name, "embedding_model_name")
        if not urls:
            return 0
        
        # Use model-aware collection
        col = await self.collection_manager.get_collection(CollectionNames.WEB_SEARCH, embedding_model_name)
        
        filt = (
            wvc.query.Filter.by_property("thread_id").equal(thread_id)
            & wvc.query.Filter.by_property("url").contains_any(urls)
        )
        try:
            await asyncio.to_thread(col.data.delete_many, where=filt)
            logger.info(f"Deleted web chunks for {len(urls)} URLs in thread '{thread_id}'")
            return len(urls)
        except WeaviateBaseError as e:
            logger.error(f"Failed to delete web chunks by URLs: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error deleting web chunks by URLs: {e}")
            return 0

    async def delete_document_vectors_by_file_hash_and_model(self, file_hash: str, embedding_model_name: str) -> bool:
        """Delete all document vectors for a file hash and embedding model.
        
        Args:
            file_hash: File hash to delete.
            embedding_model_name: Embedding model name.
            
        Returns:
            bool: True if deletion succeeded, False otherwise.
        """
        _validate_not_empty(file_hash, "file_hash")
        _validate_not_empty(embedding_model_name, "embedding_model_name")
        
        # Use model-aware collection - no need for embed_model filter
        col = await self.collection_manager.get_collection(CollectionNames.DOCUMENT, embedding_model_name)
        
        filt = wvc.query.Filter.by_property("file_hash").equal(file_hash)
        try:
            await asyncio.to_thread(col.data.delete_many, where=filt)
            logger.info(f"Deleted document vectors for file '{file_hash}', model '{embedding_model_name}'")
            return True
        except WeaviateBaseError as e:
            logger.error(f"Failed to delete document vectors: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting document vectors: {e}")
            return False

    async def has_file_indexed(self, thread_id: str, file_hash: str, embedding_model_name: str) -> bool:
        """Return whether document chunks exist for a file hash + embedding model.
        
        Args:
            thread_id: Thread identifier (for logging/context).
            file_hash: File hash to check.
            embedding_model_name: Embedding model name.
            
        Returns:
            bool: True if file is indexed, False otherwise.
            
        Raises:
            ValueError: If validation fails.
            VectorDBQueryError: If query fails.
        """
        _validate_not_empty(thread_id, "thread_id")
        _validate_not_empty(file_hash, "file_hash")
        _validate_not_empty(embedding_model_name, "embedding_model_name")
        
        # Use model-aware collection for deduplication
        try:
            col = await self.collection_manager.get_collection(CollectionNames.DOCUMENT, embedding_model_name)
            filt = (
                wvc.query.Filter.by_property("embed_model").equal(embedding_model_name)
                & wvc.query.Filter.by_property("file_hash").equal(file_hash)
            )
            response = await asyncio.to_thread(col.aggregate.over_all, filters=filt)
            has_indexed = bool(getattr(response, "total_count", 0) > 0)
            logger.debug(f"File '{file_hash}' indexed check in model-aware collection: {has_indexed}")
            return has_indexed
        except Exception as e:
            logger.error(f"Failed to check if file indexed: {e}")
            raise VectorDBQueryError("Could not check if file is indexed") from e

    async def has_chat_memory_indexed(self, thread_id: str, message_id: str) -> bool:
        """Return whether at least one chat-memory chunk exists for a message in a thread.
        
        Args:
            thread_id: Thread identifier.
            message_id: Message identifier.
            
        Returns:
            bool: True if chat memory is indexed, False otherwise.
            
        Raises:
            ValueError: If validation fails.
            VectorDBQueryError: If query fails.
        """
        _validate_not_empty(thread_id, "thread_id")
        _validate_not_empty(message_id, "message_id")
        
        # Use model-aware collection for deduplication
        try:
            # Get embedding model from thread to use correct model-aware collection
            from app.db import get_thread
            thread = await get_thread(thread_id)
            if not thread:
                return False
            
            col = await self.collection_manager.get_collection(CollectionNames.CHAT_MEMORY, thread.embed_model)
            filt = (
                wvc.query.Filter.by_property("thread_id").equal(thread_id)
                & wvc.query.Filter.by_property("message_id").equal(message_id)
            )
            response = await asyncio.to_thread(col.aggregate.over_all, filters=filt)
            has_indexed = bool(getattr(response, "total_count", 0) > 0)
            logger.debug(f"Chat memory for message '{message_id}' indexed check in model-aware collection: {has_indexed}")
            return has_indexed
        except Exception as e:
            logger.error(f"Failed to check if chat memory indexed: {e}")
            raise VectorDBQueryError("Could not check if chat memory is indexed") from e

    async def get_file_chunk_count(self, file_hash: str, embedding_model_name: str) -> int:
        """Return the number of indexed document chunks for a file/model pair.
        
        Args:
            file_hash: File hash to count chunks for.
            embedding_model_name: Embedding model name.
            
        Returns:
            int: Number of indexed chunks.
            
        Raises:
            ValueError: If validation fails.
            VectorDBQueryError: If query fails.
        """
        _validate_not_empty(file_hash, "file_hash")
        _validate_not_empty(embedding_model_name, "embedding_model_name")
        
        filt = (
            wvc.query.Filter.by_property("embed_model").equal(embedding_model_name)
            & wvc.query.Filter.by_property("file_hash").equal(file_hash)
        )
        col = self.client.collections.use(CollectionNames.DOCUMENT)
        try:
            response = await asyncio.to_thread(col.aggregate.over_all, filters=filt)
            count = int(getattr(response, "total_count", 0) or 0)
            logger.debug(f"Chunk count for file '{file_hash}': {count}")
            return count
        except WeaviateBaseError as e:
            logger.error(f"Failed to get file chunk count: {e}")
            raise VectorDBQueryError("Could not get file chunk count") from e

    async def get_thread_stats(
        self,
        thread_id: str,
        file_hashes: Optional[List[str]] = None,
        embedding_model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return aggregate vector counts for thread-scoped chat/web and filtered documents.
        
        Args:
            thread_id: Thread identifier.
            file_hashes: Optional list of file hashes to filter documents by.
            embedding_model_name: Optional embedding model name to filter by.
            
        Returns:
            Dict[str, Any]: Dictionary with counts for each collection type.
            
        Raises:
            ValueError: If validation fails.
            VectorDBQueryError: If query fails.
        """
        _validate_not_empty(thread_id, "thread_id")
        
        logger.debug(f"Getting thread stats for thread '{thread_id}'")
        
        filt = wvc.query.Filter.by_property("thread_id").equal(thread_id)

        doc_col = self.client.collections.use(CollectionNames.DOCUMENT)
        chat_col = self.client.collections.use(CollectionNames.CHAT_MEMORY)
        web_col = self.client.collections.use(CollectionNames.WEB_SEARCH)

        try:
            if file_hashes and embedding_model_name:
                doc_filter = (
                    wvc.query.Filter.by_property("embed_model").equal(embedding_model_name)
                    & wvc.query.Filter.by_property("file_hash").contains_any(file_hashes)
                )
                doc_total = await asyncio.to_thread(doc_col.aggregate.over_all, filters=doc_filter)
                pdf_count = await asyncio.to_thread(
                    doc_col.aggregate.over_all,
                    filters=doc_filter & wvc.query.Filter.by_property("source_kind").equal("pdf"),
                )
                webpage_count = await asyncio.to_thread(
                    doc_col.aggregate.over_all,
                    filters=doc_filter & wvc.query.Filter.by_property("source_kind").equal("webpage"),
                )
            else:
                doc_total = await asyncio.to_thread(doc_col.aggregate.over_all)
                pdf_count = await asyncio.to_thread(
                    doc_col.aggregate.over_all,
                    filters=wvc.query.Filter.by_property("source_kind").equal("pdf"),
                )
                webpage_count = await asyncio.to_thread(
                    doc_col.aggregate.over_all,
                    filters=wvc.query.Filter.by_property("source_kind").equal("webpage"),
                )
            chat_total = await asyncio.to_thread(chat_col.aggregate.over_all, filters=filt)
            web_total = await asyncio.to_thread(web_col.aggregate.over_all, filters=filt)
        except WeaviateBaseError as e:
            logger.error(f"Failed to get thread stats: {e}")
            raise VectorDBQueryError("Could not get thread stats") from e

        knowledge_sources = int(getattr(doc_total, "total_count", 0) or 0)
        chat_memories = int(getattr(chat_total, "total_count", 0) or 0)
        web_search_chunks = int(getattr(web_total, "total_count", 0) or 0)
        pdf_chunks = int(getattr(pdf_count, "total_count", 0) or 0)
        webpage_chunks = int(getattr(webpage_count, "total_count", 0) or 0)

        stats = {
            "exists": knowledge_sources + chat_memories + web_search_chunks > 0,
            "knowledge_sources": knowledge_sources,
            "pdf_chunks": pdf_chunks,
            "webpage_chunks": webpage_chunks,
            "chat_memories": chat_memories,
            "web_search_chunks": web_search_chunks,
            "total_points": knowledge_sources + chat_memories + web_search_chunks,
        }
        
        logger.debug(f"Thread stats for '{thread_id}': {stats}")
        return stats
