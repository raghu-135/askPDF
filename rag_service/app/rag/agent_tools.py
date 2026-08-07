from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from app.agent.tool_contract import ToolWarningCode, make_tool_error_result, make_tool_result, tool_started
from app.db import FileSourceType, ProcessStatus
from app.db.vector import get_vector_db
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET, get_embedding_model
from app.models.retry import invoke_with_retry
from app.rag.retrieval import (
    fetch_semantic_history,
    get_document_metadata_lookup,
    group_document_chunks,
    rerank_document_chunks,
)
from app.rag.enums import (
    ThreadTimelineOrder,
    ThreadTimelineSource,
    TimelineEventType,
    TimelineSourceType,
)
from app.time_utils import parse_datetime_utc


logger = logging.getLogger(__name__)

THREAD_TIMELINE_SOURCES = {source.value for source in ThreadTimelineSource}
THREAD_TIMELINE_ORDERS = {order.value for order in ThreadTimelineOrder}


class ThreadTimelineSearchInput(BaseModel):
    """Input schema for timeline-aware thread retrieval."""

    query: str = Field(
        description="Topic, entity, or temporal question to locate on the thread timeline."
    )
    sources: str = Field(
        default=ThreadTimelineSource.ALL.value,
        description="Timeline source to search: all, conversation, documents, or web_cache.",
        json_schema_extra={"enum": [source.value for source in ThreadTimelineSource]},
    )
    order: str = Field(
        default=ThreadTimelineOrder.RELEVANCE.value,
        description="Sort mode. Use oldest/newest for first/latest/before/after questions.",
        json_schema_extra={"enum": [order.value for order in ThreadTimelineOrder]},
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=30,
        description="Maximum number of timeline events to return.",
    )


class FocusedDocumentSearchInput(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    file_hash: str = Field(min_length=1, max_length=256)
    max_results: int = Field(default=10, ge=1, le=30)


def _short_excerpt(text: str, limit: int = 260) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def _event_sort_key(event: Dict[str, Any], order: str) -> Any:
    parsed = parse_datetime_utc(event.get("timeline_event_at"))
    missing_time = parsed is None
    if order == ThreadTimelineOrder.OLDEST.value:
        return (missing_time, parsed or datetime.max.replace(tzinfo=timezone.utc))
    if order == ThreadTimelineOrder.NEWEST.value:
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
    source_type = meta.get("source_type") or FileSourceType.PDF.value
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
        "source_type": TimelineSourceType.DOCUMENT.value,
        "timeline_event_at": available_at,
        "timeline_event_type": TimelineEventType.DOCUMENT_ADDED_TO_THREAD.value,
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
        "source_type": TimelineSourceType.CONVERSATION.value,
        "timeline_event_at": message_created_at,
        "timeline_event_type": TimelineEventType.MESSAGE_CREATED.value,
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
        "source_type": TimelineSourceType.WEB_CACHE.value,
        "timeline_event_at": performed_at,
        "timeline_event_type": TimelineEventType.WEB_SEARCH_PERFORMED.value,
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
    started = tool_started()
    tool_name = "get_thread_shape"
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("app_thread_id") or conf.get("thread_id")
        if not thread_id:
            return make_tool_result(
                tool_name=tool_name,
                content="No thread context found.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.MISSING_THREAD_ID],
            ).to_json()

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
                status = meta.get("indexing_status", ProcessStatus.UNKNOWN.value)
                chunks = meta.get("chunk_count", 0)
                chars = meta.get("total_chars", 0)
                words = meta.get("word_count")
                pages = meta.get("page_count")
                sentences = meta.get("sentence_count")
                name = meta.get("file_name", fh)
                stype = meta.get("source_type", FileSourceType.PDF.value)
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

        return make_tool_result(
            tool_name=tool_name,
            content="\n".join(lines),
            config=config,
            started=started,
            artifacts={"thread_shape": shape},
        ).to_json()
    except Exception as e:
        return make_tool_error_result(
            tool_name=tool_name,
            error=e,
            config=config,
            started=started,
            user_message=f"Error reading thread shape: {e}",
        ).to_json()


@tool
async def search_documents(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """
    Semantic search across all uploaded documents and cached web results.
    Returns labeled passages with surrounding context for citation.
    """
    started = tool_started()
    tool_name = "search_documents"
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("app_thread_id") or conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        context_window = conf.get("context_window", DEFAULT_TOKEN_BUDGET)
        use_reranker = conf.get("use_reranker", True)

        if not thread_id or not embedding_model:
            return make_tool_result(
                tool_name=tool_name,
                content="No thread context found.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT],
            ).to_json()

        embedding_client = get_embedding_model(embedding_model)
        query_vector = await invoke_with_retry(embedding_client.aembed_query, query)

        db = get_vector_db()
        document_lookup = await get_document_metadata_lookup(thread_id)
        thread_file_hashes = list(document_lookup.keys())
        if not thread_file_hashes:
            return make_tool_result(
                tool_name=tool_name,
                content="No documents are linked to this thread yet.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.NO_THREAD_DOCUMENTS],
            ).to_json()

        raw_doc_chunks = await db.search_knowledge_sources(
            thread_id=thread_id,
            query_vector=query_vector,
            embedding_model=embedding_model,
            limit=max_results,
            file_hashes=thread_file_hashes,
            query_text=query,
        )
        if not raw_doc_chunks:
            logger.error(
                "Missing document vectors for thread %s (files=%d, embedding_model=%s). Open thread endpoint should trigger recovery.",
                thread_id,
                len(thread_file_hashes),
                embedding_model,
            )
            return make_tool_result(
                tool_name=tool_name,
                content="Document index is missing for this thread. Re-open the thread to trigger re-indexing.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.MISSING_DOCUMENT_VECTORS],
            ).to_json()
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
                embedding_model=embedding_model,
                file_hash=file_hash,
                chunk_ids=list(id_set),
            )
            expanded_doc_chunks.extend(expanded_batch)

        expanded_doc_chunks.sort(key=lambda x: (x.get("file_hash", ""), x.get("chunk_id", 0)))

        web_chunks = await db.search_web_chunks(
            thread_id=thread_id,
            query_vector=query_vector,
            embedding_model=embedding_model,
            limit=max(3, max_results // 3),
            query_text=query,
        )
        if use_reranker:
            web_chunks = await rerank_document_chunks(query, web_chunks)

        if not expanded_doc_chunks and not web_chunks:
            return make_tool_result(
                tool_name=tool_name,
                content="No relevant content found in documents or cached web results.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.NO_RELEVANT_CONTENT],
            ).to_json()

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

        content = "\n\n".join(context_parts)
        artifacts = {
            "document_sources": document_sources,
            "web_sources": web_sources,
        }
        legacy_fields: Dict[str, Any] = {}
        if document_sources:
            legacy_fields["__document_sources__"] = document_sources
        if web_sources:
            legacy_fields["__web_sources__"] = web_sources
        return make_tool_result(
            tool_name=tool_name,
            content=content,
            config=config,
            started=started,
            sources=[*document_sources, *web_sources],
            artifacts=artifacts,
        ).to_json(legacy_fields=legacy_fields)
    except Exception as e:
        logger.error("Error in search_documents: %s", e, exc_info=True)
        return make_tool_error_result(
            tool_name=tool_name,
            error=e,
            config=config,
            started=started,
            user_message=f"Error retrieving knowledge: {e}",
        ).to_json()


@tool(args_schema=FocusedDocumentSearchInput)
async def search_document_by_id(
    query: str,
    file_hash: str,
    max_results: int = 10,
    config: RunnableConfig = None,
) -> str:
    """Semantic search inside one document that is already linked to the current thread."""

    started = tool_started()
    tool_name = "search_document_by_id"
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("app_thread_id") or conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        context_window = conf.get("context_window", DEFAULT_TOKEN_BUDGET)
        use_reranker = conf.get("use_reranker", True)
        if not thread_id or not embedding_model:
            return make_tool_result(
                tool_name=tool_name,
                content="No thread context found.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT],
            ).to_json()

        document_lookup = await get_document_metadata_lookup(thread_id)
        if file_hash not in document_lookup:
            return make_tool_result(
                tool_name=tool_name,
                content="The requested document is not linked to this thread.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.NO_THREAD_DOCUMENTS],
            ).to_json()

        embedding_client = get_embedding_model(embedding_model)
        query_vector = await invoke_with_retry(embedding_client.aembed_query, query)
        db = get_vector_db()
        raw_chunks = await db.search_knowledge_sources(
            thread_id=thread_id,
            query_vector=query_vector,
            embedding_model=embedding_model,
            limit=max_results,
            file_hashes=[file_hash],
            query_text=query,
        )
        if not raw_chunks:
            return make_tool_result(
                tool_name=tool_name,
                content="No relevant content was found in the requested document.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.NO_RELEVANT_CONTENT],
            ).to_json()
        if use_reranker:
            raw_chunks = await rerank_document_chunks(query, raw_chunks)

        expansion_radius = max(2, min(10, int(context_window / 8000) + 1))
        chunk_ids: set[int] = set()
        for hit in raw_chunks:
            chunk_id = hit.get("chunk_id")
            if chunk_id is None:
                continue
            for neighbor_id in range(int(chunk_id) - expansion_radius, int(chunk_id) + expansion_radius + 1):
                if neighbor_id >= 0:
                    chunk_ids.add(neighbor_id)
        expanded = await db.get_knowledge_source_chunks_by_ids(
            thread_id=thread_id,
            embedding_model=embedding_model,
            file_hash=file_hash,
            chunk_ids=sorted(chunk_ids),
        )
        expanded.sort(key=lambda item: int(item.get("chunk_id") or 0))
        content, sources = group_document_chunks(expanded, document_lookup)
        if not content:
            return make_tool_result(
                tool_name=tool_name,
                content="No relevant content was found in the requested document.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.NO_RELEVANT_CONTENT],
            ).to_json()
        return make_tool_result(
            tool_name=tool_name,
            content=content,
            config=config,
            started=started,
            sources=sources,
            artifacts={"document_sources": sources},
        ).to_json(legacy_fields={"__document_sources__": sources})
    except Exception as exc:
        return make_tool_error_result(
            tool_name=tool_name,
            error=exc,
            config=config,
            started=started,
            user_message=f"Error retrieving the requested document: {exc}",
        ).to_json()


@tool
async def search_thread_conversation_history(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """
    Semantic search across past conversation Q/A pairs in this thread.
    Returns the most relevant exchanges regardless of time.
    """
    started = tool_started()
    tool_name = "search_thread_conversation_history"
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("app_thread_id") or conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        use_reranker = conf.get("use_reranker", True)

        if not thread_id or not embedding_model:
            return make_tool_result(
                tool_name=tool_name,
                content="No thread context found.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT],
            ).to_json()

        embedding_client = get_embedding_model(embedding_model)
        query_vector = await invoke_with_retry(embedding_client.aembed_query, query)
        history, used_ids = await fetch_semantic_history(
            thread_id=thread_id,
            query_vector=query_vector,
            query_text=query,
            limit=max_results,
            use_reranker=use_reranker,
            embedding_model=embedding_model,
        )

        if not history:
            return make_tool_result(
                tool_name=tool_name,
                content="No relevant past conversations found.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.NO_RELEVANT_CONVERSATION_HISTORY],
            ).to_json()

        return make_tool_result(
            tool_name=tool_name,
            content=history,
            config=config,
            started=started,
            artifacts={"used_chat_ids": used_ids},
        ).to_json(legacy_fields={"__used_chat_ids__": used_ids})
    except Exception as e:
        logger.error("Error in search_thread_conversation_history: %s", e, exc_info=True)
        return make_tool_error_result(
            tool_name=tool_name,
            error=e,
            config=config,
            started=started,
            user_message=f"Error retrieving chat memory: {e}",
        ).to_json()


@tool
async def search_durable_memory(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """
    Policy-scoped search across durable user/project/thread memories.
    Returns active app-owned memory records, not raw chat history.
    """
    started = tool_started()
    tool_name = "search_durable_memory"
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("app_thread_id") or conf.get("thread_id")
        if not thread_id:
            return make_tool_result(
                tool_name=tool_name,
                content="No thread context found.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT],
            ).to_json()

        from app.models.memory_tools import MEMORY_READ_EFFECTIVE, MemorySearchInput
        from app.services.memory_tool_service import build_memory_tool_context, search_memory_tool

        memory_context, _thread, _project = await build_memory_tool_context(
            selected_scope_type="thread",
            selected_scope_id=thread_id,
            thread_id=thread_id,
            capabilities=[MEMORY_READ_EFFECTIVE],
        )
        prefetched = conf.get("prefetched_durable_memories")
        prefetch_debug = conf.get("prefetched_durable_memory_debug") or {}
        prefetch_policy = conf.get("prefetched_durable_memory_scope_policy") or {}
        recall_eligible = bool(prefetch_policy.get("searched_scopes"))
        budget_rejected = int((prefetch_debug.get("rejection_reasons") or {}).get("budget_exhausted") or 0)
        should_expand = isinstance(prefetched, list) and recall_eligible and (len(prefetched) < 3 or budget_rejected > 0)
        if isinstance(prefetched, list) and not should_expand:
            result = {
                "memories": prefetched,
                "scopes": conf.get("prefetched_durable_memory_scopes") or [],
                "scope_policy": conf.get("prefetched_durable_memory_scope_policy") or {},
                "retrieval_debug": {**prefetch_debug, "expanded_search": False, "reused_prefetch": True},
            }
        else:
            from app.services.memory_retrieval_policy import compute_memory_retrieval_budget

            expanded_budget = compute_memory_retrieval_budget(
                int(conf.get("context_window") or DEFAULT_TOKEN_BUDGET),
                expanded=True,
            )
            result = await search_memory_tool(
                memory_context,
                MemorySearchInput(
                    query=query,
                    view="effective",
                    max_results=int(expanded_budget["candidate_limit"]),
                    char_budget=int(expanded_budget["char_budget"]),
                ),
                query_vector=conf.get("prefetched_durable_memory_query_vector"),
            )
            result["retrieval_debug"] = {
                **(result.get("retrieval_debug") or {}),
                "expanded_search": True,
                "reused_prefetch": False,
            }
        memories = result.get("memories", []) if isinstance(result, dict) else []
        memory_scopes = result.get("scopes", []) if isinstance(result, dict) else []
        memory_scope_policy = result.get("scope_policy", {}) if isinstance(result, dict) else {}
        applied_overrides = result.get("applied_overrides", []) if isinstance(result, dict) else []
        suppressed_memory_ids = result.get("suppressed_memory_ids", []) if isinstance(result, dict) else []
        retrieval_debug = result.get("retrieval_debug", {}) if isinstance(result, dict) else {}

        if not memories:
            return make_tool_result(
                tool_name=tool_name,
                content="No relevant long-term memories found.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.NO_RELEVANT_MEMORY],
                artifacts={
                    "memory_refs": [],
                    "memory_scopes": memory_scopes,
                    "memory_scope_policy": memory_scope_policy,
                    "memory_applied_overrides": applied_overrides,
                    "memory_suppressed_ids": suppressed_memory_ids,
                    "memory_retrieval_debug": retrieval_debug,
                },
            ).to_json()

        lines = [
            "[LONG-TERM MEMORY]",
            "These memories are defaults. Explicit instructions in the current user request override them for this run. "
            "Otherwise Thread overrides Project, and Project overrides Personal memory.",
        ]
        memory_refs = []
        scopes = []
        for index, item in enumerate(memories, start=1):
            memory = item if isinstance(item, dict) else {}
            scope_type = memory.get("scope_type") or "unknown"
            scope_id = memory.get("scope_id") or "unknown"
            content = memory.get("excerpt") or memory.get("content") or ""
            attributes = memory.get("attributes") or {}
            lines.append(f"{index}. {scope_type}:{scope_id} ({attributes.get('kind', 'fact')}): {content}")
            memory_refs.append({
                "memory_id": memory.get("id"),
                "scope_type": scope_type,
                "scope_id": scope_id,
                "score": memory.get("score"),
                "score_type": memory.get("score_type"),
                "raw_score": memory.get("raw_score"),
                "embedding_model": memory.get("embedding_model"),
                "scope_rank": memory.get("scope_rank"),
                "attributes": attributes,
            })
            scope_key = {"scope_type": scope_type, "scope_id": scope_id}
            if scope_key not in scopes:
                scopes.append(scope_key)

        return make_tool_result(
            tool_name=tool_name,
            content="\n".join(lines),
            config=config,
            started=started,
            artifacts={
                "memory_refs": memory_refs,
                "memory_scopes": memory_scopes or scopes,
                "memory_scope_policy": memory_scope_policy,
                "memory_applied_overrides": applied_overrides,
                "memory_suppressed_ids": suppressed_memory_ids,
                "memory_precedence": result.get("precedence", ["thread", "project", "user"]),
                "memory_representation_issues": result.get("representation_issues", []),
                "memory_retrieval_debug": retrieval_debug,
            },
        ).to_json()
    except Exception as e:
        logger.error("Error in search_durable_memory: %s", e, exc_info=True)
        return make_tool_error_result(
            tool_name=tool_name,
            error=e,
            config=config,
            started=started,
            user_message=f"Error retrieving long-term memory: {e}",
        ).to_json()


@tool(args_schema=ThreadTimelineSearchInput)
async def search_thread_events(
    query: str,
    sources: ThreadTimelineSource | str = ThreadTimelineSource.ALL.value,
    order: ThreadTimelineOrder | str = ThreadTimelineOrder.RELEVANCE.value,
    max_results: int = 10,
    config: RunnableConfig = None,
) -> str:
    """
    Search timestamped events in the current thread timeline.

    Use this tool when the user asks about chronology, sequence, recency,
    earliest/latest evidence, what happened before or after another event, or
    what changed since a time.
    """
    started = tool_started()
    tool_name = "search_thread_events"
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("app_thread_id") or conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        use_reranker = conf.get("use_reranker", True)
        if not thread_id or not embedding_model:
            return make_tool_result(
                tool_name=tool_name,
                content="No thread context found.",
                config=config,
                started=started,
                warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT],
            ).to_json()

        max_results = max(1, min(int(max_results or 10), 30))
        source_value = sources.value if isinstance(sources, ThreadTimelineSource) else str(sources)
        order_value = order.value if isinstance(order, ThreadTimelineOrder) else str(order)
        requested_sources = source_value if source_value in THREAD_TIMELINE_SOURCES else ThreadTimelineSource.ALL.value
        order_value = order_value if order_value in THREAD_TIMELINE_ORDERS else ThreadTimelineOrder.RELEVANCE.value
        db = get_vector_db()
        events: List[Dict[str, Any]] = []

        needs_vector = requested_sources in {
            ThreadTimelineSource.ALL.value,
            ThreadTimelineSource.CONVERSATION.value,
            ThreadTimelineSource.WEB_CACHE.value,
        }
        query_vector: Optional[List[float]] = None
        if needs_vector:
            embedding_client = get_embedding_model(embedding_model)
            query_vector = await invoke_with_retry(embedding_client.aembed_query, query)

        if requested_sources in {ThreadTimelineSource.ALL.value, ThreadTimelineSource.CONVERSATION.value} and query_vector is not None:
            recalled = await db.search_chat_memory(
                thread_id=thread_id,
                query_vector=query_vector,
                embedding_model=embedding_model,
                limit=max_results,
            )
            if use_reranker and query:
                recalled = await rerank_document_chunks(query, recalled)
            for mem in recalled:
                event = _chat_timeline_event(mem)
                if event:
                    events.append(event)

        if requested_sources in {ThreadTimelineSource.ALL.value, ThreadTimelineSource.DOCUMENTS.value}:
            document_lookup = await get_document_metadata_lookup(thread_id)
            for file_hash, meta in document_lookup.items():
                if not isinstance(meta, dict):
                    continue
                event = _document_timeline_event(file_hash, meta)
                if event:
                    events.append(event)

        if requested_sources in {ThreadTimelineSource.ALL.value, ThreadTimelineSource.WEB_CACHE.value} and query_vector is not None:
            web_chunks = await db.search_web_chunks(
                thread_id=thread_id,
                query_vector=query_vector,
                embedding_model=embedding_model,
                limit=max_results,
                query_text=query,
            )
            if use_reranker and query:
                web_chunks = await rerank_document_chunks(query, web_chunks)
            for chunk in web_chunks:
                event = _web_timeline_event(chunk)
                if event:
                    events.append(event)

        events.sort(key=lambda event: _event_sort_key(event, order_value))
        events = events[:max_results]

        return make_tool_result(
            tool_name=tool_name,
            content=_format_timeline_content(events),
            config=config,
            started=started,
            sources=events,
            artifacts={"timeline_events": events},
            warnings=[] if events else [ToolWarningCode.NO_TIMELINE_EVENTS],
        ).to_json(legacy_fields={"__timeline_events__": events})
    except Exception as e:
        logger.error("Error in search_thread_events: %s", e, exc_info=True)
        return make_tool_error_result(
            tool_name=tool_name,
            error=e,
            config=config,
            started=started,
            user_message=f"Error searching thread timeline: {e}",
        ).to_json()
