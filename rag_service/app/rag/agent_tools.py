from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.db.vector import get_vector_db
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET, get_embedding_model
from app.models.retry import invoke_with_retry
from app.rag.retrieval import (
    fetch_semantic_history,
    get_document_metadata_lookup,
    group_document_chunks,
    rerank_document_chunks,
)
from app.time_utils import parse_datetime_utc


logger = logging.getLogger(__name__)


class ThreadTimelineSearchInput(BaseModel):
    """Input schema for timeline-aware thread retrieval."""

    query: str = Field(
        description="Topic, entity, or temporal question to locate on the thread timeline."
    )
    sources: Literal["all", "conversation", "documents", "web_cache"] = Field(
        default="all",
        description="Timeline source to search: all, conversation, documents, or web_cache.",
    )
    order: Literal["relevance", "oldest", "newest"] = Field(
        default="relevance",
        description="Sort mode. Use oldest/newest for first/latest/before/after questions.",
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Maximum number of timeline events to return.",
    )


def _short_excerpt(text: str, limit: int = 260) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def _event_sort_key(event: Dict[str, Any], order: str) -> Any:
    parsed = parse_datetime_utc(event.get("timeline_event_at"))
    missing_time = parsed is None
    if order == "oldest":
        return (missing_time, parsed or datetime.max.replace(tzinfo=timezone.utc))
    if order == "newest":
        oldest = datetime.min.replace(tzinfo=timezone.utc)
        return (missing_time, -(parsed or oldest).timestamp())
    try:
        score_value = float(event.get("score") or 0.0)
    except Exception:
        score_value = 0.0
    newest = parsed.timestamp() if parsed else float("-inf")
    return (-score_value, -newest)


def _format_timeline_content(events: List[Dict[str, Any]]) -> str:
    if not events:
        return "No timeline events matched the request."

    lines = ["[THREAD TIMELINE EVENTS]"]
    for event in events:
        at = event.get("timeline_event_at") or "unknown time"
        event_type = event.get("timeline_event_type") or "unknown_event"
        label = event.get("label") or event.get("source_type") or "source"
        excerpt = event.get("excerpt") or ""
        lines.append(f"- {at} | {event_type} | {label}: {excerpt}")
    return "\n".join(lines)


def _document_timeline_event(file_hash: str, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    available_at = meta.get("document_available_in_thread_at")
    if not available_at:
        return None
    file_name = meta.get("file_name") or file_hash
    source_type = meta.get("source_type") or "pdf"
    details = []
    for label, field in (
        ("pages", "page_count"),
        ("words", "word_count"),
        ("sentences", "sentence_count"),
    ):
        value = meta.get(field)
        if value not in (None, ""):
            details.append(f"{value} {label}")
    detail_text = f" ({', '.join(details)})" if details else ""
    return {
        "source_type": "document",
        "timeline_event_at": available_at,
        "timeline_event_type": "document_added_to_thread",
        "document_available_in_thread_at": available_at,
        "file_hash": file_hash,
        "file_name": file_name,
        "document_source_type": source_type,
        "label": f"Document added to thread: {file_name}",
        "excerpt": f"{file_name} was added to this thread{detail_text}.",
        "page_count": meta.get("page_count"),
        "word_count": meta.get("word_count"),
        "sentence_count": meta.get("sentence_count"),
        "languages": meta.get("languages"),
        "filetype": meta.get("filetype"),
        "element_types": meta.get("element_types"),
    }


def _chat_timeline_event(mem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    message_created_at = mem.get("message_created_at")
    if not message_created_at:
        return None
    score = mem.get("rerank_score", mem.get("score"))
    event: Dict[str, Any] = {
        "source_type": "conversation",
        "timeline_event_at": message_created_at,
        "timeline_event_type": "message_created",
        "message_created_at": message_created_at,
        "message_id": mem.get("message_id"),
        "label": "Conversation memory",
        "excerpt": _short_excerpt(mem.get("text", "")),
    }
    if score is not None:
        event["score"] = score
    return event


def _web_timeline_event(chunk: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    performed_at = chunk.get("web_search_performed_at")
    if not performed_at:
        return None
    title = chunk.get("title") or "Internet Search"
    url = chunk.get("url") or ""
    label = f'Cached web result: "{title}"'
    if url:
        label += f" | {url}"
    score = chunk.get("rerank_score", chunk.get("score"))
    event: Dict[str, Any] = {
        "source_type": "web_cache",
        "timeline_event_at": performed_at,
        "timeline_event_type": "web_search_performed",
        "web_search_performed_at": performed_at,
        "url": url,
        "title": title,
        "search_query": chunk.get("search_query"),
        "label": label,
        "excerpt": _short_excerpt(chunk.get("text", "")),
    }
    if score is not None:
        event["score"] = score
    return event


@tool
async def get_thread_shape(config: RunnableConfig = None) -> str:
    """
    Snapshot of thread content inventory: documents + QA history volume.
    Use to calibrate retrieval strategy before making tool calls.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        if not thread_id:
            return "No thread context found."

        from app.db import get_thread_shape as _get_shape
        shape = await _get_shape(thread_id)

        qa_pairs = shape["total_qa_pairs"]
        avg_qa = shape["avg_qa_chars"]
        total_qa = shape["total_qa_chars"]
        docs = shape["documents"]

        lines = ["[THREAD SHAPE]"]
        lines.append(
            f"QA History  : {qa_pairs} pair(s) | {avg_qa:,.0f} avg chars/pair | {total_qa:,} total chars"
        )
        if docs:
            lines.append(f"Documents   : {len(docs)} source(s)")
            for i, (fh, meta) in enumerate(docs.items(), start=1):
                status = meta.get("indexing_status", "unknown")
                chunks = meta.get("chunk_count", 0)
                chars = meta.get("total_chars", 0)
                words = meta.get("word_count")
                pages = meta.get("page_count")
                sentences = meta.get("sentence_count")
                name = meta.get("file_name", fh)
                stype = meta.get("source_type", "pdf")
                available_at = meta.get("document_available_in_thread_at")
                doc_counts = []
                if pages not in (None, ""):
                    doc_counts.append(f"{pages} pages")
                if words not in (None, ""):
                    doc_counts.append(f"{words:,} words")
                if sentences not in (None, ""):
                    doc_counts.append(f"{sentences:,} sentences")
                counts_text = f" | {', '.join(doc_counts)}" if doc_counts else ""
                availability = f" | added_to_thread_at={available_at}" if available_at else ""
                lines.append(
                    f"  {i}. file_name={name} | file_hash={fh} | source_type={stype} | "
                    f"{chunks} chunks | {chars:,} chars{counts_text} | {status}{availability}"
                )
        else:
            lines.append("Documents   : none uploaded yet")

        return "\n".join(lines)
    except Exception as e:
        return f"Error reading thread shape: {e}"


@tool
async def search_documents(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """
    Semantic search across all uploaded documents and cached web results.
    Returns labeled passages with surrounding context for citation.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        context_window = conf.get("context_window", DEFAULT_TOKEN_BUDGET)
        use_reranker = conf.get("use_reranker", True)

        if not thread_id or not embedding_model:
            return "No thread context found."

        embed_model = get_embedding_model(embedding_model)
        query_vector = await invoke_with_retry(embed_model.aembed_query, query)

        db = get_vector_db()
        document_lookup = await get_document_metadata_lookup(thread_id)
        thread_file_hashes = list(document_lookup.keys())
        if not thread_file_hashes:
            return "No documents are linked to this thread yet."

        raw_doc_chunks = await db.search_knowledge_sources(
            thread_id=thread_id,
            query_vector=query_vector,
            embedding_model_name=embedding_model,
            limit=max_results,
            file_hashes=thread_file_hashes,
            query_text=query,
        )
        if not raw_doc_chunks:
            logger.error(
                "Missing document vectors for thread %s (files=%d, embed_model=%s). Open thread endpoint should trigger recovery.",
                thread_id,
                len(thread_file_hashes),
                embedding_model,
            )
            return "Document index is missing for this thread. Re-open the thread to trigger re-indexing."
        if use_reranker:
            raw_doc_chunks = await rerank_document_chunks(query, raw_doc_chunks)

        expansion_radius = max(2, min(10, int(context_window / 8000) + 1))
        file_chunk_map: Dict[str, set[int]] = {}
        for hit in raw_doc_chunks:
            file_hash = hit.get("file_hash")
            chunk_id = hit.get("chunk_id")
            if file_hash is not None and chunk_id is not None:
                file_chunk_map.setdefault(file_hash, set())
                for neighbor_id in range(chunk_id - expansion_radius, chunk_id + expansion_radius + 1):
                    if neighbor_id >= 0:
                        file_chunk_map[file_hash].add(neighbor_id)

        expanded_doc_chunks = []
        for file_hash, id_set in file_chunk_map.items():
            expanded_batch = await db.get_knowledge_source_chunks_by_ids(
                thread_id=thread_id,
                embedding_model_name=embedding_model,
                file_hash=file_hash,
                chunk_ids=list(id_set),
            )
            expanded_doc_chunks.extend(expanded_batch)

        expanded_doc_chunks.sort(key=lambda x: (x.get("file_hash", ""), x.get("chunk_id", 0)))

        web_chunks = await db.search_web_chunks(
            thread_id=thread_id,
            query_vector=query_vector,
            embedding_model_name=embedding_model,
            limit=max(3, max_results // 3),
            query_text=query,
        )
        if use_reranker:
            web_chunks = await rerank_document_chunks(query, web_chunks)

        if not expanded_doc_chunks and not web_chunks:
            return "No relevant content found in documents or cached web results."

        context_parts = []
        document_context, document_sources = group_document_chunks(expanded_doc_chunks, document_lookup)
        if document_context:
            context_parts.append(document_context)

        web_sources = []
        web_groups: Dict[str, Dict[str, Any]] = {}
        for wchunk in web_chunks:
            url = wchunk.get("url", "")
            performed_at = wchunk.get("web_search_performed_at")
            web_groups.setdefault(
                url,
                {
                    "title": wchunk.get("title", url),
                    "texts": [],
                    "web_search_performed_at": performed_at,
                },
            )
            web_groups[url]["texts"].append(wchunk.get("text", ""))

            score = wchunk.get("rerank_score", wchunk.get("score", 0.0))
            web_source: Dict[str, Any] = {
                "text": wchunk.get("text", "")[:200] + "...",
                "url": url,
                "title": wchunk.get("title", url),
                "score": score,
            }
            for field in ("web_search_performed_at", "timeline_event_at", "timeline_event_type"):
                value = wchunk.get(field)
                if value not in (None, ""):
                    web_source[field] = value
            web_sources.append(web_source)

        for url, group in web_groups.items():
            combined_text = "\n".join(group["texts"])
            performed_at = group.get("web_search_performed_at")
            prefix = f"Cached web result from search performed at {performed_at}:\n" if performed_at else ""
            context_parts.append(f'{prefix}[Source: Internet Search - "{group["title"]}" | {url}]\n{combined_text}')

        result: Dict[str, Any] = {"content": "\n\n".join(context_parts)}
        if document_sources:
            result["__document_sources__"] = document_sources
        if web_sources:
            result["__web_sources__"] = web_sources
        return json.dumps(result)
    except Exception as e:
        logger.error("Error in search_documents: %s", e, exc_info=True)
        return f"Error retrieving knowledge: {e}"


@tool
async def search_conversation_history(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """
    Semantic search across past conversation Q/A pairs in this thread.
    Returns the most relevant exchanges regardless of time.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        use_reranker = conf.get("use_reranker", True)

        if not thread_id or not embedding_model:
            return "No thread context found."

        embed_model = get_embedding_model(embedding_model)
        query_vector = await invoke_with_retry(embed_model.aembed_query, query)
        history, used_ids = await fetch_semantic_history(
            thread_id=thread_id,
            query_vector=query_vector,
            query_text=query,
            limit=max_results,
            use_reranker=use_reranker,
            embedding_model_name=embedding_model,
        )

        if not history:
            return "No relevant past conversations found."

        return json.dumps({
            "content": history,
            "__used_chat_ids__": used_ids,
        })
    except Exception as e:
        return f"Error retrieving chat memory: {e}"


@tool(args_schema=ThreadTimelineSearchInput)
async def search_thread_timeline(
    query: str,
    sources: Literal["all", "conversation", "documents", "web_cache"] = "all",
    order: Literal["relevance", "oldest", "newest"] = "relevance",
    max_results: int = 10,
    config: RunnableConfig = None,
) -> str:
    """
    Search timestamped events in the current thread timeline.

    Use this tool when the user asks about chronology, sequence, recency,
    earliest/latest evidence, what happened before or after another event, or
    what changed since a time.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        use_reranker = conf.get("use_reranker", True)
        if not thread_id or not embedding_model:
            return "No thread context found."

        max_results = max(1, min(int(max_results or 10), 30))
        requested_sources = sources if sources in {"all", "conversation", "documents", "web_cache"} else "all"
        order = order if order in {"relevance", "oldest", "newest"} else "relevance"
        db = get_vector_db()
        events: List[Dict[str, Any]] = []

        needs_vector = requested_sources in {"all", "conversation", "web_cache"}
        query_vector: Optional[List[float]] = None
        if needs_vector:
            embed_model = get_embedding_model(embedding_model)
            query_vector = await invoke_with_retry(embed_model.aembed_query, query)

        if requested_sources in {"all", "conversation"} and query_vector is not None:
            recalled = await db.search_chat_memory(
                thread_id=thread_id,
                query_vector=query_vector,
                embedding_model_name=embedding_model,
                limit=max_results,
            )
            if use_reranker and query:
                recalled = await rerank_document_chunks(query, recalled)
            for mem in recalled:
                event = _chat_timeline_event(mem)
                if event:
                    events.append(event)

        if requested_sources in {"all", "documents"}:
            document_lookup = await get_document_metadata_lookup(thread_id)
            for file_hash, meta in document_lookup.items():
                if not isinstance(meta, dict):
                    continue
                event = _document_timeline_event(file_hash, meta)
                if event:
                    events.append(event)

        if requested_sources in {"all", "web_cache"} and query_vector is not None:
            web_chunks = await db.search_web_chunks(
                thread_id=thread_id,
                query_vector=query_vector,
                embedding_model_name=embedding_model,
                limit=max_results,
                query_text=query,
            )
            if use_reranker and query:
                web_chunks = await rerank_document_chunks(query, web_chunks)
            for chunk in web_chunks:
                event = _web_timeline_event(chunk)
                if event:
                    events.append(event)

        events.sort(key=lambda event: _event_sort_key(event, order))
        events = events[:max_results]

        return json.dumps({
            "content": _format_timeline_content(events),
            "__timeline_events__": events,
        })
    except Exception as e:
        logger.error("Error in search_thread_timeline: %s", e, exc_info=True)
        return f"Error searching thread timeline: {e}"
