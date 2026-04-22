"""
vector_db.py - Weaviate vector database adapter with global separated collections

Collections:
- DocumentChunk   : PDF and indexed webpage chunks
- ChatMemoryChunk : semantic memory chunks from QA pairs
- WebSearchChunk  : cached internet search snippets
"""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import weaviate
import weaviate.classes as wvc


DOCUMENT_COLLECTION = "DocumentChunk"
CHAT_COLLECTION = "ChatMemoryChunk"
WEB_SEARCH_COLLECTION = "WebSearchChunk"

_singleton_instance: Optional["WeaviateAdapter"] = None


def get_vector_db() -> "WeaviateAdapter":
    """Return a shared Weaviate adapter singleton."""
    global _singleton_instance
    if _singleton_instance is None:
        _singleton_instance = WeaviateAdapter()
    return _singleton_instance


class WeaviateAdapter:
    """Adapter for Weaviate with thread-filtered global collections."""

    def __init__(self) -> None:
        """Initialize Weaviate client connection and ensure required collections exist."""
        weaviate_url = os.getenv("WEAVIATE_URL")
        if not weaviate_url:
            raise ValueError("WEAVIATE_URL environment variable is not set")

        self.hybrid_alpha = float(os.getenv("WEAVIATE_HYBRID_ALPHA", "0.5"))
        parsed = urlparse(weaviate_url if "://" in weaviate_url else f"http://{weaviate_url}")
        host = parsed.hostname or "weaviate"
        port = parsed.port or 8080
        secure = parsed.scheme == "https"

        self.client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=secure,
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=secure,
            skip_init_checks=True,
        )

        self._ensure_collections_sync()

    async def ensure_collections(self) -> None:
        """Ensure all required Weaviate collections exist (async wrapper)."""
        await asyncio.to_thread(self._ensure_collections_sync)

    def _ensure_collections_sync(self) -> None:
        """Synchronously create all application collections if missing."""
        self._ensure_collection(
            DOCUMENT_COLLECTION,
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
            CHAT_COLLECTION,
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
            WEB_SEARCH_COLLECTION,
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

    def _ensure_collection(self, name: str, properties: List[tuple[str, Any]]) -> None:
        """Create a single collection with self-provided vectors when absent."""
        if self.client.collections.exists(name):
            return
        self.client.collections.create(
            name=name,
            vector_config=wvc.config.Configure.Vectors.self_provided(),
            properties=[
                wvc.config.Property(name=prop_name, data_type=prop_dtype)
                for prop_name, prop_dtype in properties
            ],
        )

    @staticmethod
    def _metadata_json(metadata: Optional[Dict[str, Any]]) -> str:
        """Serialize metadata dict to JSON; return `{}` on empty/invalid input."""
        if not metadata:
            return "{}"
        import json

        try:
            return json.dumps(metadata)
        except Exception:
            return "{}"

    @staticmethod
    def _parse_metadata(raw: Optional[str]) -> Dict[str, Any]:
        """Parse metadata JSON string into a dict; return empty dict on failure."""
        if not raw:
            return {}
        import json

        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _score(obj: Any) -> float:
        """Extract retrieval score (or distance fallback) from Weaviate result metadata."""
        meta = getattr(obj, "metadata", None)
        if not meta:
            return 0.0
        score = getattr(meta, "score", None)
        if score is None:
            score = getattr(meta, "distance", None)
        if score is None:
            return 0.0
        try:
            return float(score)
        except Exception:
            return 0.0

    async def _insert_many(self, collection_name: str, points: List[Dict[str, Any]]) -> int:
        """Insert vector points into a target collection and return inserted count."""
        if not points:
            return 0
        col = self.client.collections.use(collection_name)
        for p in points:
            await asyncio.to_thread(
                col.data.insert,
                properties=p["properties"],
                vector=p["vector"],
                uuid=str(uuid.uuid4()),
            )
        return len(points)

    async def delete_thread_data(self, thread_id: str) -> bool:
        """Delete only thread-scoped chat-memory and web-search vectors for a thread."""
        filt = wvc.query.Filter.by_property("thread_id").equal(thread_id)
        try:
            for name in [CHAT_COLLECTION, WEB_SEARCH_COLLECTION]:
                col = self.client.collections.use(name)
                await asyncio.to_thread(col.data.delete_many, where=filt)
            return True
        except Exception:
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
        """Index PDF chunks into `DocumentChunk` with embedding-model/file identifiers."""
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
                        "metadata_json": self._metadata_json(md),
                    },
                }
            )
        return await self._insert_many(DOCUMENT_COLLECTION, points)

    async def index_chat_memory(
        self,
        thread_id: str,
        message_id: str,
        question: str,
        answer: str,
        texts: List[str],
        embeddings: List[List[float]],
    ) -> int:
        """Index compact chat-memory chunks into `ChatMemoryChunk`."""
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
        return await self._insert_many(CHAT_COLLECTION, points)

    async def index_web_search_chunks(
        self,
        thread_id: str,
        query: str,
        texts: List[str],
        embeddings: List[List[float]],
        urls: Optional[List[str]] = None,
        titles: Optional[List[str]] = None,
    ) -> int:
        """Index transient web-search snippets into `WebSearchChunk`."""
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
        return await self._insert_many(WEB_SEARCH_COLLECTION, points)

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
        """Search document chunks using vector or hybrid retrieval with file/model filters."""
        base_filter = wvc.query.Filter.by_property("embed_model").equal(embedding_model_name)
        if file_hash:
            base_filter = base_filter & wvc.query.Filter.by_property("file_hash").equal(file_hash)
        if file_hashes:
            base_filter = base_filter & wvc.query.Filter.by_property("file_hash").contains_any(file_hashes)

        col = self.client.collections.use(DOCUMENT_COLLECTION)
        kwargs = {
            "filters": base_filter,
            "limit": limit,
            "return_metadata": wvc.query.MetadataQuery(score=True),
        }

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
                    "score": self._score(obj),
                    "metadata": self._parse_metadata(p.get("metadata_json")),
                }
            )
        return results

    async def get_knowledge_source_chunks_by_ids(
        self,
        thread_id: str,
        embedding_model_name: str,
        file_hash: str,
        chunk_ids: List[int],
    ) -> List[Dict[str, Any]]:
        """Fetch specific document chunks by file hash and chunk IDs."""
        if not chunk_ids:
            return []
        filt = (
            wvc.query.Filter.by_property("embed_model").equal(embedding_model_name)
            & wvc.query.Filter.by_property("file_hash").equal(file_hash)
            & wvc.query.Filter.by_property("chunk_id").contains_any(chunk_ids)
        )

        col = self.client.collections.use(DOCUMENT_COLLECTION)
        response = await asyncio.to_thread(
            col.query.fetch_objects,
            filters=filt,
            limit=len(chunk_ids),
            return_metadata=wvc.query.MetadataQuery(score=True),
        )

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
                    "metadata": self._parse_metadata(p.get("metadata_json")),
                }
            )

        out.sort(key=lambda x: x.get("chunk_id", 0))
        return out

    async def search_chat_memory(
        self,
        thread_id: str,
        query_vector: List[float],
        limit: int = 3,
        exclude_message_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search chat-memory vectors within a thread, with optional message exclusions."""
        filt = wvc.query.Filter.by_property("thread_id").equal(thread_id)
        if exclude_message_ids:
            filt = filt & wvc.query.Filter.not_(
                wvc.query.Filter.by_property("message_id").contains_any(exclude_message_ids)
            )

        col = self.client.collections.use(CHAT_COLLECTION)
        response = await asyncio.to_thread(
            col.query.near_vector,
            near_vector=query_vector,
            filters=filt,
            limit=limit,
            return_metadata=wvc.query.MetadataQuery(score=True),
        )

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
                    "score": self._score(obj),
                }
            )
        return out

    async def search_web_chunks(
        self,
        thread_id: str,
        query_vector: List[float],
        limit: int = 5,
        query_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search web-search snippet vectors for a thread via vector or hybrid query."""
        filt = wvc.query.Filter.by_property("thread_id").equal(thread_id)
        col = self.client.collections.use(WEB_SEARCH_COLLECTION)

        kwargs = {
            "filters": filt,
            "limit": limit,
            "return_metadata": wvc.query.MetadataQuery(score=True),
        }

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
                    "score": self._score(obj),
                }
            )
        return out

    async def delete_chat_memory_by_message_id(self, thread_id: str, message_id: str) -> bool:
        """Delete all chat-memory chunks belonging to a single assistant message."""
        filt = (
            wvc.query.Filter.by_property("thread_id").equal(thread_id)
            & wvc.query.Filter.by_property("message_id").equal(message_id)
        )
        try:
            col = self.client.collections.use(CHAT_COLLECTION)
            await asyncio.to_thread(col.data.delete_many, where=filt)
            return True
        except Exception:
            return False

    async def delete_web_chunks_by_urls(self, thread_id: str, urls: List[str]) -> int:
        """Delete web-search chunks for a thread whose URLs match any provided URL."""
        if not urls:
            return 0
        filt = (
            wvc.query.Filter.by_property("thread_id").equal(thread_id)
            & wvc.query.Filter.by_property("url").contains_any(urls)
        )
        try:
            col = self.client.collections.use(WEB_SEARCH_COLLECTION)
            await asyncio.to_thread(col.data.delete_many, where=filt)
            return len(urls)
        except Exception:
            return 0

    async def delete_document_vectors_by_file_hash_and_model(self, file_hash: str, embedding_model_name: str) -> bool:
        """Delete all document vectors for a file hash and embedding model."""
        filt = (
            wvc.query.Filter.by_property("embed_model").equal(embedding_model_name)
            & wvc.query.Filter.by_property("file_hash").equal(file_hash)
        )
        try:
            col = self.client.collections.use(DOCUMENT_COLLECTION)
            await asyncio.to_thread(col.data.delete_many, where=filt)
            return True
        except Exception:
            return False

    async def has_file_indexed(self, thread_id: str, file_hash: str, embedding_model_name: str) -> bool:
        """Return whether document chunks exist for a file hash + embedding model."""
        filt = (
            wvc.query.Filter.by_property("embed_model").equal(embedding_model_name)
            & wvc.query.Filter.by_property("file_hash").equal(file_hash)
        )
        col = self.client.collections.use(DOCUMENT_COLLECTION)
        response = await asyncio.to_thread(col.aggregate.over_all, filters=filt)
        return bool(getattr(response, "total_count", 0) > 0)

    async def has_chat_memory_indexed(self, thread_id: str, message_id: str) -> bool:
        """Return whether at least one chat-memory chunk exists for a message in a thread."""
        filt = (
            wvc.query.Filter.by_property("thread_id").equal(thread_id)
            & wvc.query.Filter.by_property("message_id").equal(message_id)
        )
        col = self.client.collections.use(CHAT_COLLECTION)
        response = await asyncio.to_thread(col.aggregate.over_all, filters=filt)
        return bool(getattr(response, "total_count", 0) > 0)

    async def get_file_chunk_count(self, file_hash: str, embedding_model_name: str) -> int:
        """Return the number of indexed document chunks for a file/model pair."""
        filt = (
            wvc.query.Filter.by_property("embed_model").equal(embedding_model_name)
            & wvc.query.Filter.by_property("file_hash").equal(file_hash)
        )
        col = self.client.collections.use(DOCUMENT_COLLECTION)
        response = await asyncio.to_thread(col.aggregate.over_all, filters=filt)
        return int(getattr(response, "total_count", 0) or 0)

    async def get_thread_stats(
        self,
        thread_id: str,
        file_hashes: Optional[List[str]] = None,
        embedding_model_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return aggregate vector counts for thread-scoped chat/web and filtered documents."""
        filt = wvc.query.Filter.by_property("thread_id").equal(thread_id)

        doc_col = self.client.collections.use(DOCUMENT_COLLECTION)
        chat_col = self.client.collections.use(CHAT_COLLECTION)
        web_col = self.client.collections.use(WEB_SEARCH_COLLECTION)

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

        knowledge_sources = int(getattr(doc_total, "total_count", 0) or 0)
        chat_memories = int(getattr(chat_total, "total_count", 0) or 0)
        web_search_chunks = int(getattr(web_total, "total_count", 0) or 0)
        pdf_chunks = int(getattr(pdf_count, "total_count", 0) or 0)
        webpage_chunks = int(getattr(webpage_count, "total_count", 0) or 0)

        return {
            "exists": knowledge_sources + chat_memories + web_search_chunks > 0,
            "knowledge_sources": knowledge_sources,
            "pdf_chunks": pdf_chunks,
            "webpage_chunks": webpage_chunks,
            "chat_memories": chat_memories,
            "web_search_chunks": web_search_chunks,
            "total_points": knowledge_sources + chat_memories + web_search_chunks,
        }
