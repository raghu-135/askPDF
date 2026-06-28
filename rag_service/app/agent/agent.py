import asyncio
import logging
import json
import time
import re
from datetime import datetime, timezone
from typing import TypedDict, List, Annotated, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from app.models.llm_server_client import get_llm, get_embedding_model, DEFAULT_TOKEN_BUDGET, DEFAULT_MAX_ITERATIONS
from app.prompts.loaders import (
    get_orchestrator_prompt,
    get_orchestrator_prompt_compact,
    get_orchestrator_phase0_prompt,
    get_orchestrator_phase0_prompt_compact,
    get_intent_agent_prompt,
    get_web_search_mandate,
)
from app.agent.agent_helpers import (
    build_chat_prompt,
    parse_intent_response,
    evidence_insufficient,
    collect_tool_sources,
    format_runtime_datetime_context,
)
from app.agent.external_research_tools import (
    get_external_research_tools,
    search_web,
    search_web_intent,
)
from app.prompts.defaults import DEFAULT_SYSTEM_ROLE
from app.db import is_file_in_thread
from app.db.vector import get_vector_db
from app.rag.retrieval import (
    fetch_semantic_history,
    get_document_metadata_lookup,
    group_document_chunks,
    rerank_document_chunks,
)
from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG
from app.time_utils import parse_datetime_utc

logger = logging.getLogger(__name__)


def _extract_http_status_code(err_str: str) -> Optional[int]:
    patterns = (
        r"status(?:_code)?[=:]\s*(\d{3})",
        r"error code:\s*(\d{3})",
        r"\b(\d{3})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, err_str)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def _is_retryable_model_error(err_str: str) -> tuple[bool, str]:
    status_code = _extract_http_status_code(err_str)
    if status_code in {408, 409, 429} or (status_code is not None and status_code >= 500):
        return True, f"Retryable OpenAI-compatible API error ({status_code})"
    return False, ""


async def invoke_with_retry(func, *args, **kwargs):
    max_retries = 10
    base_delay = 2
    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_retryable, reason = _is_retryable_model_error(err_str)
            if is_retryable:
                delay = base_delay * (2 ** min(i, 4)) # Exponential backoff up to 32s max delay
                logger.warning(f"{reason}. Retrying in {delay}s... (Attempt {i+1}/{max_retries})")
                await asyncio.sleep(delay)
                continue
            raise
    raise Exception("Max retries reached while waiting for model to become available.")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    llm_model: str
    embedding_model: str
    context_window: int
    use_web_search: bool
    document_sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    used_chat_ids: List[str]
    clarification_options: Optional[List[str]]
    iteration_count: int
    max_iterations: int
    system_role: str
    tool_instructions: Dict[str, str]
    custom_instructions: str
    pre_fetch_bundle: Optional[Dict[str, Any]]
    intent_agent_ran: bool  # True when Intent Agent preprocessed the query; False = Orchestrator self-preprocesses
    reasoning_mode: bool
    working_query: str
    intent_reference_type: str
    client_timezone: Optional[str]
    client_locale: Optional[str]
    client_now_iso: Optional[str]


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
async def search_documents(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """
    Semantic search across all uploaded documents and cached web results.
    Returns labeled passages with surrounding context for citation.

    Args:
        query: Natural-language question or description of the fact to locate.
        max_results: Number of seed chunks to retrieve before context expansion.
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

        # ── Build a file_hash → file_name lookup from thread metadata (no DB join) ──
        document_lookup = await get_document_metadata_lookup(thread_id)
        thread_file_hashes = list(document_lookup.keys())
        if not thread_file_hashes:
            return "No documents are linked to this thread yet."

        # ── Document chunk search with neighbor expansion ──
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
                "Missing document vectors for thread %s (files=%d, embed_model=%s). "
                "Open thread endpoint should trigger recovery.",
                thread_id,
                len(thread_file_hashes),
                embedding_model,
            )
            return "Document index is missing for this thread. Re-open the thread to trigger re-indexing."
        if use_reranker:
            raw_doc_chunks = await rerank_document_chunks(query, raw_doc_chunks)

        expansion_radius = max(2, min(10, int(context_window / 8000) + 1))
        file_chunk_map = {}
        for hit in raw_doc_chunks:
            file_hash = hit.get("file_hash")
            chunk_id = hit.get("chunk_id")
            if file_hash is not None and chunk_id is not None:
                if file_hash not in file_chunk_map:
                    file_chunk_map[file_hash] = set()
                for neighbor_id in range(chunk_id - expansion_radius, chunk_id + expansion_radius + 1):
                    if neighbor_id >= 0:
                        file_chunk_map[file_hash].add(neighbor_id)

        expanded_doc_chunks = []
        for file_hash, id_set in file_chunk_map.items():
            expanded_batch = await db.get_knowledge_source_chunks_by_ids(
                thread_id=thread_id,
                embedding_model_name=embedding_model,
                file_hash=file_hash,
                chunk_ids=list(id_set)
            )
            expanded_doc_chunks.extend(expanded_batch)

        expanded_doc_chunks.sort(key=lambda x: (x.get("file_hash", ""), x.get("chunk_id", 0)))

        # ── Cached web search results ──
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

        document_sources = []
        web_sources = []
        context_parts = []

        # Group document chunks by file to reduce context window bloat
        document_context, document_sources = group_document_chunks(expanded_doc_chunks, document_lookup)
        if document_context:
            context_parts.append(document_context)

        # Group cached web chunks by URL
        web_groups = {}
        for wchunk in web_chunks:
            url = wchunk.get("url", "")
            performed_at = wchunk.get("web_search_performed_at")
            if url not in web_groups:
                web_groups[url] = {
                    "title": wchunk.get("title", url),
                    "texts": [],
                    "web_search_performed_at": performed_at,
                }
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
            context_parts.append(f'{prefix}[Source: Internet Search — "{group["title"]}" | {url}]\n{combined_text}')

        result: Dict[str, Any] = {"content": "\n\n".join(context_parts)}
        if document_sources:
            result["__document_sources__"] = document_sources
        if web_sources:
            result["__web_sources__"] = web_sources
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error in search_documents: {e}", exc_info=True)
        return f"Error retrieving knowledge: {e}"


@tool
async def search_conversation_history(query: str, max_results: int = 10, config: RunnableConfig = None) -> str:
    """
    Semantic search across past conversation Q/A pairs in this thread.
    Returns the most relevant exchanges regardless of time.

    Args:
        query: Natural-language description of the topic or fact to recall.
        max_results: Maximum number of past QA pairs to return.
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
            "__used_chat_ids__": used_ids
        })
    except Exception as e:
        return f"Error retrieving chat memory: {e}"


@tool
async def ask_for_clarification(options: List[str]) -> str:
    """
    Present the user with 2–4 distinct interpretations of their question.
    Each option must be a complete, self-contained question.

    Args:
        options: List of 2–4 complete questions representing distinct interpretations.
    """
    return json.dumps({"__clarification_options__": options})




@tool
async def search_document_by_id(
    query: str,
    file_hash: str,
    max_results: int = 8,
    config: RunnableConfig = None,
) -> str:
    """
    Semantic search within a single document identified by file_hash.

    Args:
        query: Natural-language question to search for within the document.
        file_hash: File hash of the target document.
        max_results: Number of seed chunks before neighbor expansion.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        thread_id = conf.get("thread_id")
        embedding_model = conf.get("embedding_model")
        context_window = conf.get("context_window", DEFAULT_TOKEN_BUDGET)

        if not thread_id or not embedding_model:
            return "No thread context found."
        if not await is_file_in_thread(thread_id, file_hash):
            return f"Document {file_hash} is not linked to this thread."

        embed_model = get_embedding_model(embedding_model)
        query_vector = await invoke_with_retry(embed_model.aembed_query, query)

        db = get_vector_db()
        raw_chunks = await db.search_knowledge_sources(
            thread_id=thread_id,
            query_vector=query_vector,
            embedding_model_name=embedding_model,
            limit=max_results,
            file_hash=file_hash,
            query_text=query,
        )

        if not raw_chunks:
            logger.error(
                "Missing vectors for file %s in thread %s (embed_model=%s).",
                file_hash,
                thread_id,
                embedding_model,
            )
            return f"No relevant content found in document {file_hash}."

        expansion_radius = max(2, min(10, int(context_window / 8000) + 1))
        chunk_ids_to_fetch: set = set()
        for hit in raw_chunks:
            chunk_id = hit.get("chunk_id")
            if chunk_id is not None:
                for neighbor_id in range(chunk_id - expansion_radius, chunk_id + expansion_radius + 1):
                    if neighbor_id >= 0:
                        chunk_ids_to_fetch.add(neighbor_id)

        expanded_chunks = await db.get_knowledge_source_chunks_by_ids(
            thread_id=thread_id,
            embedding_model_name=embedding_model,
            file_hash=file_hash,
            chunk_ids=list(chunk_ids_to_fetch),
        )
        expanded_chunks.sort(key=lambda x: x.get("chunk_id", 0))

        # Resolve file name for source attribution from thread metadata (no DB join)
        document_lookup = await get_document_metadata_lookup(thread_id)
        fname = document_lookup.get(file_hash, {}).get("file_name", file_hash)

        document_context, sources = group_document_chunks(
            expanded_chunks,
            {file_hash: document_lookup.get(file_hash, {"file_name": fname})},
        )
        if not document_context:
            document_context = ""

        return json.dumps({
            "content": document_context,
            "__document_sources__": sources,
        })
    except Exception as e:
        logger.error(f"Error in search_document_by_id: {e}", exc_info=True)
        return f"Error searching document: {e}"


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
    what changed since a time. Do not use it for ordinary semantic document
    search when ordering does not matter; use search_documents,
    search_document_by_id, or search_conversation_history for that.

    Timestamps have source-specific meanings. message_created_at is when a
    conversation memory message was stored. document_available_in_thread_at is
    when a document was added to this thread, not when the file was
    globally created or when the document was authored. web_search_performed_at
    is when cached web evidence was fetched, not when the page was published.

    Args:
        query: Topic, entity, or temporal question to locate on the timeline.
        sources: One of all, conversation, documents, or web_cache.
        order: Sort events by relevance, oldest, or newest.
        max_results: Maximum number of timeline events to return.
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
        logger.error(f"Error in search_thread_timeline: {e}", exc_info=True)
        return f"Error searching thread timeline: {e}"


# ---------------------------------------------------------------------------
# Pre-fetch bundle formatter
# ---------------------------------------------------------------------------

def _format_prefetch_for_prompt(bundle: Optional[Dict[str, Any]]) -> str:
    """
    Format a pre-fetched context bundle as a labelled block for injection into
    LLM system prompts.  Returns an empty string when the bundle is None/empty.
    """
    if not bundle:
        return ""

    parts: List[str] = []

    stats = bundle.get("stats", {})
    if stats and stats.get("total_messages", 0) > 0:
        parts.append(
            f"[CONVERSATION STATS]  Total turns: {stats.get('total_messages', 0)} | "
            f"Estimated tokens in history: {int(stats.get('estimated_history_tokens', 0))}"
        )

    docs = bundle.get("documents", [])
    if docs:
        doc_lines = []
        for d in docs:
            source_type = d.get("source_type") or "unknown"
            available_at = d.get("document_available_in_thread_at") or "unknown"
            counts = []
            if d.get("page_count") not in (None, ""):
                counts.append(f"pages: {d['page_count']}")
            if d.get("word_count") not in (None, ""):
                counts.append(f"words: {d['word_count']}")
            if d.get("sentence_count") not in (None, ""):
                counts.append(f"sentences: {d['sentence_count']}")
            if d.get("chunk_count") not in (None, ""):
                counts.append(f"chunks: {d['chunk_count']}")
            counts_text = f" | {' | '.join(counts)}" if counts else ""
            doc_lines.append(
                f"  {d['index']}. {d['file_name']} "
                f"(file_hash: {d['file_hash']} | source_type: {source_type} | "
                f"added_to_thread_at: {available_at}{counts_text})"
            )
        doc_lines = "\n".join(doc_lines)
        parts.append(f"[UPLOADED DOCUMENTS — {len(docs)} file(s)]\n{doc_lines}")

    recent = bundle.get("recent_history_text", "")
    if recent:
        parts.append(f"[RECENT CONVERSATION TURNS]\n{recent}")

    semantic = bundle.get("semantic_history_text", "")
    if semantic:
        parts.append(f"[SEMANTICALLY RELEVANT PAST QA PAIRS]\n{semantic}")

    documents = bundle.get("document_evidence_text", "")
    if documents:
        parts.append(f"[DOCUMENT EVIDENCE (PDF + WEBPAGE)  (queried with raw/un-rewritten question)]\n{documents}")

    web = bundle.get("web_evidence_text", "")
    if web:
        parts.append(f"[WEB SEARCH EVIDENCE]\n{web}")

    if not parts:
        return ""

    sep = "=" * 64
    return (
        f"\n\n{sep}\n"
        "PRE-FETCHED CONTEXT  (assembled before this call — no tool calls needed for this data):\n"
        f"{sep}\n"
        + "\n\n".join(parts)
        + f"\n{sep}\n"
        "NOTE: Document Evidence and Semantic History were retrieved with the raw question.\n"
        "A better-rewritten query will improve precision — call tools ONLY when this\n"
        "pre-fetched context is genuinely insufficient to answer the user's request.\n"
        f"{sep}"
    )


def format_orchestrator_tool_context(use_external_research: bool) -> str:
    """Summarize the downstream Orchestrator's active tools for intent handoff."""
    catalog = get_tool_catalog(get_active_tools(use_external_research=use_external_research))
    if not catalog:
        return ""

    lines = [
        "## DOWNSTREAM ORCHESTRATOR TOOL CATALOG (NOT CALLABLE BY YOU)",
        "",
        "These tools are available only to the Orchestrator after it receives your rewritten query.",
        "Do not call these tools from the Intent Agent; use this catalog only to preserve source, connector, and tool constraints in the rewritten query:",
    ]
    for item in catalog:
        lines.append(
            f"- `{item['tool_name']}` ({item['display_name']}): {item['description']}"
        )
    return "\n".join(lines)


def format_intent_tool_context(active_intent_tools: List[Any]) -> str:
    """Summarize tools actually bound to the Intent Agent."""
    if not active_intent_tools:
        return "\n".join(
            [
                "## INTENT AGENT TOOL CATALOG (CALLABLE BY YOU NOW)",
                "",
                "No intent-stage tools are active for this session.",
            ]
        )

    catalog = get_tool_catalog(active_intent_tools)
    lines = [
        "## INTENT AGENT TOOL CATALOG (CALLABLE BY YOU NOW)",
        "",
        "Only the tools in this section are callable by you before submitting your route:",
    ]
    for item in catalog:
        lines.append(
            f"- `{item['tool_name']}` ({item['display_name']}): {item['description']} "
            f"Guidance: {item['default_prompt']}"
        )
    return "\n".join(lines)


# --- Intent Agent ---
class IntentAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    thread_id: str
    llm_model: str
    context_window: int
    iteration_count: int
    max_iterations: int
    intent_result: Optional[Dict[str, Any]]
    pre_fetch_bundle: Optional[Dict[str, Any]]
    reasoning_mode: bool
    intent_tools_used: bool
    use_web_search: bool
    client_timezone: Optional[str]
    client_locale: Optional[str]
    client_now_iso: Optional[str]



async def call_intent_model(state: IntentAgentState, config: RunnableConfig):
    """
    Single-pass Intent Agent: classifies and rewrites the user's query for the Orchestrator.
    No tools, no retries — one LLM call, one JSON output.
    """
    messages = state["messages"]
    llm = get_llm(state["llm_model"], temperature=0.0)
    allow_intent_web_search = bool(state.get("use_web_search"))
    
    tools_to_bind = intent_tools.copy() if allow_intent_web_search else []
    
    if tools_to_bind:
        llm_with_tools = llm.bind_tools(tools_to_bind)
    else:
        llm_with_tools = llm
    iteration = state.get("iteration_count", 0) + 1
    context_window = state.get("context_window", DEFAULT_TOKEN_BUDGET)
    reasoning_mode = state.get("reasoning_mode", True)

    # Load and format the Intent Agent prompt
    bundle = state.get("pre_fetch_bundle")
    prefetch_text = _format_prefetch_for_prompt(bundle) if bundle else ""

    base_prompt = get_intent_agent_prompt()
    intent_tool_context = format_intent_tool_context(tools_to_bind)
    orchestrator_tool_context = format_orchestrator_tool_context(
        use_external_research=allow_intent_web_search
    )
    system_prompt = (
        base_prompt
        .replace("{INTENT_TOOL_CONTEXT}", intent_tool_context)
        .replace("{ORCHESTRATOR_TOOL_CONTEXT}", orchestrator_tool_context)
        .replace(
            "{RUNTIME_DATETIME_CONTEXT}",
            format_runtime_datetime_context(
                client_timezone=state.get("client_timezone"),
                client_locale=state.get("client_locale"),
                client_now_iso=state.get("client_now_iso"),
            ),
        )
        .replace("{PREFETCH_CONTEXT}", prefetch_text)
    )
    if not allow_intent_web_search:
        system_prompt += "\n\nWeb search is disabled for this session. Do NOT call any web search tools."
    prompt_template = build_chat_prompt()
    input_messages = prompt_template.format_messages(
        system_prompt=system_prompt,
        messages=messages if not (messages and isinstance(messages[0], SystemMessage)) else messages[1:],
    )

    # Log complete prompt for Intent Agent in OpenAI-like format
    logger.debug(f"--- INTENT AGENT PROMPT BEGIN [thread_id: {state.get('thread_id')}] ---")
    payload = []
    for msg in input_messages:
        role = "system" if isinstance(msg, SystemMessage) else "user" if isinstance(msg, HumanMessage) else "assistant"
        entry = {"role": role, "content": msg.content}
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
        if isinstance(msg, ToolMessage):
            entry["role"] = "tool"
            entry["tool_call_id"] = msg.tool_call_id
        payload.append(entry)
    logger.debug(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.debug(f"--- INTENT AGENT PROMPT END ---")

    # Single direct call — no tools, minimal retries
    try:
        response = await invoke_with_retry(llm_with_tools.ainvoke, input_messages)
    except Exception as e:
        logger.error(f"Intent Agent LLM call failed: {e}")
        original_question = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        intent_result = {
            "route": "ANSWER",
            "rewritten_query": original_question,
            "reference_type": "NONE",
            "context_coverage": "PARTIAL",
            "clarification_options": None,
        }
        return {"messages": [AIMessage(content=json.dumps(intent_result))], "iteration_count": iteration, "intent_result": intent_result}

    if getattr(response, "tool_calls", None):
        # If it reaches here, it's calling search_web_intent
        return {
            "messages": [response],
            "iteration_count": iteration,
            "intent_result": None,
        }

    intent_result = parse_intent_response(response.content, logger=logger)
    if intent_result is None:
        logger.warning("Intent XML invalid; retrying once with strict XML instruction.")
        retry_msg = HumanMessage(
            content=(
                "Your previous output was invalid or missing required XML tags. Output the required XML tags: "
                "<route>, <rewritten_query>, <reference_type>, <context_coverage>. If route is CLARIFY, include "
                "2–4 <option> questions written as direct, standalone, first-person questions spoken by the user."
            )
        )
        retry_messages = input_messages + [retry_msg]
        try:
            retry_response = await invoke_with_retry(llm_with_tools.ainvoke, retry_messages)
            intent_result = parse_intent_response(retry_response.content, logger=logger)
            response = retry_response
        except Exception as e:
            logger.error(f"Intent Agent retry failed: {e}")

    if intent_result is None:
        original_question = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        intent_result = {
            "route": "ANSWER",
            "rewritten_query": original_question,
            "reference_type": "NONE",
            "context_coverage": "INSUFFICIENT",
            "clarification_options": None,
        }
        logger.warning("Intent XML invalid after retry; falling back to original question.")

    return {
        "messages": [response],
        "iteration_count": iteration,
        "intent_result": intent_result,
    }




class IntentToolNode(ToolNode):
    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        tool_calls = _log_tool_invocation_start("intent_tools", input, config)
        start = time.perf_counter()
        try:
            res = await super().ainvoke(input, config, **kwargs)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "Tool invocation failed | node=%s elapsed_ms=%.1f calls=%s",
                "intent_tools",
                elapsed_ms,
                _truncate_for_log(tool_calls),
            )
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        _log_tool_invocation_end("intent_tools", tool_calls, res, elapsed_ms, config, input)
        return {
            "messages": res.get("messages", []),
            "intent_tools_used": True,
        }


def intent_should_continue(state: IntentAgentState):
    messages = state["messages"]
    last_message = messages[-1]
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 1)
    tools_used = state.get("intent_tools_used", False)

    if getattr(last_message, "tool_calls", None):
        if tools_used and iteration_count >= max_iterations:
            return END
        return "tools"
    return END


intent_tools = [
    search_web_intent,
]

intent_workflow = StateGraph(IntentAgentState)
intent_workflow.add_node("agent", call_intent_model)
intent_workflow.add_node("tools", IntentToolNode(intent_tools))
intent_workflow.add_edge(START, "agent")
intent_workflow.add_conditional_edges(
    "agent",
    intent_should_continue,
    {"tools": "tools", END: END},
)
intent_workflow.add_edge("tools", "agent")

intent_app = intent_workflow.compile()
# --- End Intent Agent ---

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
                availability = (
                    f" | added_to_thread_at={available_at}"
                    if available_at else ""
                )
                lines.append(
                    f"  {i}. file_name={name} | file_hash={fh} | source_type={stype} | "
                    f"{chunks} chunks | {chars:,} chars{counts_text} | {status}{availability}"
                )
        else:
            lines.append("Documents   : none uploaded yet")

        return "\n".join(lines)
    except Exception as e:
        return f"Error reading thread shape: {e}"


core_tools_list = [
    get_thread_shape,
    search_documents,
    search_document_by_id,
    search_conversation_history,
    search_thread_timeline,
    search_web,
    ask_for_clarification,
]

external_research_tools = get_external_research_tools()

tools_list = [
    *core_tools_list,
    *external_research_tools,
]


def get_active_tools(use_external_research: bool) -> List[Any]:
    if use_external_research:
        return tools_list
    return core_tools_list


def _truncate_for_log(value: Any, max_chars: int = 500) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    if len(text) > max_chars:
        return text[:max_chars] + "...[truncated]"
    return text


def _extract_tool_calls_for_log(input_state: dict) -> List[Dict[str, Any]]:
    messages = input_state.get("messages") or []
    if not messages:
        return []
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    summary = []
    for call in tool_calls:
        if isinstance(call, dict):
            summary.append(
                {
                    "id": call.get("id"),
                    "name": call.get("name"),
                    "args": call.get("args", call.get("parameters")),
                }
            )
        else:
            summary.append(
                {
                    "id": getattr(call, "id", None),
                    "name": getattr(call, "name", None),
                    "args": getattr(call, "args", None),
                }
            )
    return summary


def _log_tool_invocation_start(
    node_name: str,
    input_state: dict,
    config: Optional[RunnableConfig],
) -> List[Dict[str, Any]]:
    tool_calls = _extract_tool_calls_for_log(input_state)
    conf = config.get("configurable", {}) if config else {}
    thread_id = conf.get("thread_id", input_state.get("thread_id"))
    for call in tool_calls:
        logger.info(
            "Tool invocation start | node=%s thread_id=%s tool=%s call_id=%s args=%s",
            node_name,
            thread_id,
            call.get("name"),
            call.get("id"),
            _truncate_for_log(call.get("args")),
        )
    return tool_calls


def _log_tool_invocation_end(
    node_name: str,
    tool_calls: List[Dict[str, Any]],
    result: dict,
    elapsed_ms: float,
    config: Optional[RunnableConfig],
    input_state: dict,
) -> None:
    conf = config.get("configurable", {}) if config else {}
    thread_id = conf.get("thread_id", input_state.get("thread_id"))
    messages = result.get("messages", []) if isinstance(result, dict) else []
    tool_result_index = 0
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        call = tool_calls[tool_result_index] if tool_result_index < len(tool_calls) else {}
        tool_result_index += 1
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        logger.info(
            "Tool invocation end | node=%s thread_id=%s tool=%s call_id=%s elapsed_ms=%.1f result_chars=%d result_preview=%s",
            node_name,
            thread_id,
            call.get("name"),
            getattr(msg, "tool_call_id", None) or call.get("id"),
            elapsed_ms,
            len(content),
            _truncate_for_log(content, max_chars=300),
        )


def _format_recoverable_tool_error(error: Exception) -> str:
    return (
        f"Tool execution failed: {type(error).__name__}: {error}. "
        "Treat this source as unavailable and continue with other available evidence. "
        "If this source is required to answer, explain the limitation."
    )



class OrchestratorToolNode(ToolNode):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("handle_tool_errors", _format_recoverable_tool_error)
        super().__init__(*args, **kwargs)

    async def ainvoke(self, input: dict, config: Optional[RunnableConfig] = None, **kwargs: Any) -> Any:
        # Intercept tool calls to extract special JSON state updates
        tool_calls = _log_tool_invocation_start("orchestrator_tools", input, config)
        start = time.perf_counter()
        try:
            res = await super().ainvoke(input, config, **kwargs)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "Tool invocation failed | node=%s elapsed_ms=%.1f calls=%s",
                "orchestrator_tools",
                elapsed_ms,
                _truncate_for_log(tool_calls),
            )
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        _log_tool_invocation_end("orchestrator_tools", tool_calls, res, elapsed_ms, config, input)

        document_sources = list(input.get("document_sources", []))
        web_sources = list(input.get("web_sources", []))
        used_chat_ids = list(input.get("used_chat_ids", []))
        clarification_options = None

        messages = res.get("messages", [])
        for i, msg in enumerate(messages):
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str) and msg.content.startswith("{") and "__" in msg.content:
                try:
                    data = json.loads(msg.content)
                    # Replace the raw message content with the clean text
                    if "content" in data:
                        messages[i].content = data["content"]
                    if "__document_sources__" in data:
                        document_sources.extend(data["__document_sources__"])
                    if "__web_sources__" in data:
                        web_sources.extend(data["__web_sources__"])
                    if "__used_chat_ids__" in data:
                        used_chat_ids.extend(data["__used_chat_ids__"])
                    if "__clarification_options__" in data:
                        clarification_options = data["__clarification_options__"]
                        messages[i].content = f"Interrupted for clarification with options: {clarification_options}"
                except Exception as e:
                    logger.warning(f"Failed to parse tool JSON output: {e}")

        return {
            "messages": messages,
            "document_sources": document_sources,
            "web_sources": web_sources,
            "used_chat_ids": used_chat_ids,
            "clarification_options": clarification_options,
        }

async def call_model(state: AgentState, config: RunnableConfig):
    messages = state["messages"]
    llm = get_llm(state["llm_model"])
    iteration = state.get("iteration_count", 0) + 1
    
    context_window = state.get('context_window', DEFAULT_TOKEN_BUDGET)
    system_role = sanitize_system_role(state.get("system_role", ""))
    tool_instructions = normalize_tool_instructions(state.get("tool_instructions", {}))
    custom_instructions = sanitize_custom_instructions(state.get("custom_instructions", ""))
    reasoning_mode = state.get("reasoning_mode", True)
    use_web_search = state.get("use_web_search", False)
    intent_agent_ran = state.get("intent_agent_ran", True)

    prompt_content = build_system_prompt(
        context_window=context_window,
        system_role=system_role,
        tool_instructions=tool_instructions,
        custom_instructions=custom_instructions,
        use_web_search=use_web_search,
        intent_agent_ran=intent_agent_ran,
        reasoning_mode=reasoning_mode,
        client_timezone=state.get("client_timezone"),
        client_locale=state.get("client_locale"),
        client_now_iso=state.get("client_now_iso"),
    )

    # Inject pre-fetched context bundle + pre-fetch-first retrieval policy
    bundle = state.get("pre_fetch_bundle")
    if bundle:
        bundle_text = _format_prefetch_for_prompt(bundle)
        if bundle_text:
            if reasoning_mode:
                prompt_content += (
                    "\n\nPRE-FETCH RETRIEVAL POLICY (LOCKED):\n"
                    "Pre-fetched context (recent turns, semantic history, document evidence, document list) is\n"
                    "already present in the PRE-FETCHED CONTEXT block below. Before calling any tool:\n"
                    "1. Assess whether the pre-fetched content answers the rewritten query with confidence.\n"
                    "   If YES, skip document/history tool calls — but still follow the external retrieval\n"
                    "   mandate below when active external retrieval tools are available.\n"
                    "2. If document evidence is present but the rewritten query is more specific than the raw question:\n"
                    "   call search_documents or search_document_by_id ONCE with the rewritten query.\n"
                    "3. If the question targets a specific document and its file_hash is in the document list:\n"
                    "   prefer search_document_by_id (scoped) over search_documents (all documents).\n"
                    "4. If the Intent Agent classified the reference type as TEMPORAL, prefer\n"
                    "   search_thread_timeline over search_conversation_history for first/latest/since/order questions.\n"
                    "5. Do NOT call search_conversation_history just to re-read recent turns — the recent\n"
                    "   conversation and semantic history are already in the pre-fetched block.\n"
                    + (
                        "6. EXTERNAL SEARCH IS ENABLED AND MANDATORY: call search_web for general web needs,\n"
                        "   and preserve explicit source or tool constraints from the user when selecting\n"
                        "   among the active external retrieval tools. Pre-fetched document evidence does\n"
                        "   NOT replace external search. Batch external search with any document search.\n"
                        if use_web_search else ""
                    )
                ) + bundle_text
            else:
                prompt_content += (
                    "\n\nPRE-FETCH RETRIEVAL POLICY (LOCKED):\n"
                    "Use the PRE-FETCHED CONTEXT block below directly. Tool calls are disabled in compact mode;\n"
                    "the system will run automatic retrieval after this response if evidence is insufficient."
                ) + bundle_text

    prompt_template = build_chat_prompt()
    input_messages = prompt_template.format_messages(
        system_prompt=prompt_content,
        messages=messages,
    )

    active_tools = get_active_tools(use_external_research=use_web_search)
    llm_with_tools = llm.bind_tools(active_tools) if reasoning_mode else llm

    # Log complete prompt for Orchestrator Agent in OpenAI-like format
    logger.debug(f"--- ORCHESTRATOR AGENT PROMPT BEGIN [thread_id: {state.get('thread_id')}, iteration: {iteration}] ---")
    payload = []
    for msg in input_messages:
        role = "system" if isinstance(msg, SystemMessage) else "user" if isinstance(msg, HumanMessage) else "assistant"
        entry = {"role": role, "content": msg.content}
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            entry["tool_calls"] = msg.tool_calls
        if isinstance(msg, ToolMessage):
            entry["role"] = "tool"
            entry["tool_call_id"] = msg.tool_call_id
        payload.append(entry)
    logger.debug(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.debug(f"--- ORCHESTRATOR AGENT PROMPT END ---")

    response = await invoke_with_retry(llm_with_tools.ainvoke, input_messages)
    return {"messages": [response], "iteration_count": iteration}


def _looks_like_tool_call_text(text: str) -> bool:
    if not text:
        return False
    try:
        data = json.loads(text)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if data.get("type") == "function":
        return True
    if "function" in data and "parameters" in data:
        return True
    if "tool" in data and "tool_input" in data:
        return True
    return False


async def force_final_answer(state: AgentState, config: RunnableConfig):
    """
    Fallback when the tool-iteration budget is exhausted or the model returns empty text.
    Rebuilds a clean, flat prompt from the retrieved tool content to avoid confusing the
    model with a broken tool-calling message chain (empty AIMessages, multiple ToolMessages).
    """
    messages = state["messages"]
    llm = get_llm(state["llm_model"])
    iteration = state.get("iteration_count", 0) + 1

    context_window = state.get('context_window', DEFAULT_TOKEN_BUDGET)
    system_role = sanitize_system_role(state.get("system_role", ""))
    tool_instructions = normalize_tool_instructions(state.get("tool_instructions", {}))
    custom_instructions = sanitize_custom_instructions(state.get("custom_instructions", ""))
    use_web_search = state.get("use_web_search", False)
    intent_agent_ran = state.get("intent_agent_ran", True)
    reasoning_mode = state.get("reasoning_mode", True)

    # ── Extract original user question ──
    original_question = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            original_question = msg.content if isinstance(msg.content, str) else str(msg.content)
            break  # take the first human message (the actual user question)

    # ── Collect all tool results and earlier clean AI turns ──
    tool_context_parts: list[str] = []
    prior_ai_parts: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            # ToolMessages may still be raw JSON if OrchestratorToolNode didn't strip them
            if isinstance(content, str) and content.startswith("{") and "content" in content:
                try:
                    parsed = json.loads(content)
                    content = parsed.get("content", content)
                except Exception:
                    pass
            if content.strip():
                tool_context_parts.append(content.strip())
        elif isinstance(msg, AIMessage):
            txt = msg.content if isinstance(msg.content, str) else ""
            if isinstance(msg.content, list):
                from reasoning import _text_from_content_item
                txt = "\n".join([_text_from_content_item(i) for i in msg.content if i]).strip()
            # Only keep non-empty AI turns that are not tool-call-only turns
            if txt.strip() and not getattr(msg, "tool_calls", None):
                prior_ai_parts.append(txt.strip())

    # ── Build a direct synthesis prompt ──
    sys_prompt = build_system_prompt(
        context_window=context_window,
        system_role=system_role,
        tool_instructions=tool_instructions,
        custom_instructions=custom_instructions,
        use_web_search=use_web_search,
        intent_agent_ran=intent_agent_ran,
        reasoning_mode=reasoning_mode,
        client_timezone=state.get("client_timezone"),
        client_locale=state.get("client_locale"),
        client_now_iso=state.get("client_now_iso"),
    )

    parts = []
    if tool_context_parts:
        parts.append("RETRIEVED CONTEXT (from tool searches):\n\n" + "\n\n---\n\n".join(tool_context_parts))
    if prior_ai_parts:
        parts.append("EARLIER ANALYSIS:\n\n" + "\n\n".join(prior_ai_parts))

    force_content = (
        "You MUST now write a final answer. Do NOT call any tools.\n\n"
        + ("\n\n".join(parts) + "\n\n" if parts else "")
        + f"USER QUESTION:\n{original_question}\n\n"
        "Write a complete, helpful answer based on the retrieved context above. "
        "Cite sources where available. If the context is insufficient, say so honestly."
    )
    force_msg = HumanMessage(content=force_content)

    prompt_template = build_chat_prompt()
    input_messages = prompt_template.format_messages(
        system_prompt=sys_prompt,
        messages=[force_msg],
    )
    try:
        response = await invoke_with_retry(llm.ainvoke, input_messages)
    except Exception as e:
        logger.error(f"Force final answer LLM call failed: {e}")
        # Return a static fallback message inside an AIMessage so the graph can finish gracefully
        fallback_text = (
            "I have retrieved some information but am currently unable to synthesize "
            "a final response due to a technical issue with the model server. "
            "Please try again in a few moments."
        )
        response = AIMessage(content=fallback_text)
    
    return {"messages": [response], "iteration_count": iteration}


def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", DEFAULT_MAX_ITERATIONS)
    reasoning_mode = state.get("reasoning_mode", True)

    if getattr(last_message, "tool_calls", None):
        if iteration_count >= max_iterations:
            # If the only pending call is search_web and no web search has run yet,
            # grant one extra pass so we don't force-finalize with zero web context.
            pending_tool_names = {tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "") for tc in last_message.tool_calls}
            web_sources_so_far = state.get("web_sources", [])
            if pending_tool_names == {"search_web"} and not web_sources_so_far:
                logger.info("Granting one extra iteration for search_web (no web results yet).")
                return "tools"
            logger.warning(f"Reaching max agent iterations ({iteration_count}/{max_iterations}). Forcing termination.")
            return "force_final_answer"
        return "tools"

    if not reasoning_mode and evidence_insufficient(state):
        if iteration_count >= max_iterations:
            logger.warning(f"Reaching max agent iterations ({iteration_count}/{max_iterations}). Forcing termination.")
            return "force_final_answer"
        logger.info("Non-reasoning mode: auto-tools pass triggered (no tool calls, insufficient evidence).")
        return "auto_tools"

    # Detect empty-content response after tool execution (e.g. model outputs nothing after
    # receiving tool results). Force a final answer instead of silently ending with blank text.
    if iteration_count > 0:
        content = getattr(last_message, "content", "")
        if isinstance(content, list):
            from reasoning import _text_from_content_item
            text_body = "\n".join([_text_from_content_item(i) for i in content if i]).strip()
        else:
            text_body = (content or "").strip()

        if not text_body:
            logger.warning("LLM returned empty response after tool execution. Triggering force_final_answer.")
            return "force_final_answer"

    if not reasoning_mode:
        content = getattr(last_message, "content", "")
        if isinstance(content, list):
            from reasoning import _text_from_content_item
            content = "\n".join([_text_from_content_item(i) for i in content if i]).strip()
        if isinstance(content, str) and _looks_like_tool_call_text(content.strip()):
            logger.warning("Non-reasoning mode: model returned tool-call-like JSON. Forcing final answer.")
            return "force_final_answer"

    return END


async def auto_tools(state: AgentState, config: RunnableConfig):
    """
    Non-reasoning fallback: run required tools when the model fails to call any.
    """
    tool_messages: list[ToolMessage] = []
    document_sources: list[Dict[str, Any]] = []
    web_sources: list[Dict[str, Any]] = []
    used_chat_ids: list[str] = []

    working_query = state.get("working_query", "")
    use_web_search = state.get("use_web_search", False)
    intent_ref = state.get("intent_reference_type", "NONE")
    prefetch = state.get("pre_fetch_bundle") or {}
    documents = prefetch.get("documents") or []

    async def _record_auto_tool_result(tool_name: str, result: str, elapsed_ms: float):
        tool_messages.append(ToolMessage(content=result, tool_call_id=f"auto_{tool_name}"))
        collect_tool_sources(result, document_sources, web_sources, used_chat_ids)
        logger.info(
            "Tool invocation end | node=auto_tools thread_id=%s tool=%s call_id=auto_%s elapsed_ms=%.1f result_chars=%d result_preview=%s",
            state.get("thread_id"),
            tool_name,
            tool_name,
            elapsed_ms,
            len(result or ""),
            _truncate_for_log(result or "", max_chars=300),
        )

    async def _run_auto_tool(tool_name: str, args: Dict[str, Any], tool_func):
        logger.info(
            "Tool invocation start | node=auto_tools thread_id=%s tool=%s call_id=auto_%s args=%s",
            state.get("thread_id"),
            tool_name,
            tool_name,
            _truncate_for_log(args),
        )
        start = time.perf_counter()
        try:
            result = await tool_func()
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "Tool invocation failed | node=auto_tools thread_id=%s tool=%s call_id=auto_%s elapsed_ms=%.1f args=%s",
                state.get("thread_id"),
                tool_name,
                tool_name,
                elapsed_ms,
                _truncate_for_log(args),
            )
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        await _record_auto_tool_result(tool_name, result, elapsed_ms)

    if use_web_search and not state.get("web_sources"):
        await _run_auto_tool(
            "search_web",
            {"query": working_query},
            lambda: search_web.ainvoke({"query": working_query}, config=config),
        )

    if documents and not state.get("document_sources"):
        if intent_ref == "ENTITY" and len(documents) == 1:
            file_hash = documents[0].get("file_hash")
            if file_hash:
                await _run_auto_tool(
                    "search_document_by_id",
                    {"query": working_query, "file_hash": file_hash},
                    lambda: search_document_by_id.ainvoke(
                        {"query": working_query, "file_hash": file_hash},
                        config=config,
                    ),
                )
        else:
            await _run_auto_tool(
                "search_documents",
                {"query": working_query},
                lambda: search_documents.ainvoke({"query": working_query}, config=config),
            )

    if intent_ref == "TEMPORAL":
        await _run_auto_tool(
            "search_thread_timeline",
            {"query": working_query, "sources": "all", "order": "relevance"},
            lambda: search_thread_timeline.ainvoke(
                {"query": working_query, "sources": "all", "order": "relevance"},
                config=config,
            ),
        )
    elif intent_ref == "SEMANTIC":
        await _run_auto_tool(
            "search_conversation_history",
            {"query": working_query},
            lambda: search_conversation_history.ainvoke({"query": working_query}, config=config),
        )

    return {
        "messages": tool_messages,
        "document_sources": document_sources,
        "web_sources": web_sources,
        "used_chat_ids": used_chat_ids,
    }

def clarification_router(state: AgentState):
    if state.get("clarification_options"):
        return END  # Suspend graph
    return "agent"


# Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", OrchestratorToolNode(tools_list))
workflow.add_node("force_final_answer", force_final_answer)
workflow.add_node("auto_tools", auto_tools)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", "auto_tools": "auto_tools", "force_final_answer": "force_final_answer", END: END},
)
workflow.add_edge("auto_tools", "agent")

workflow.add_conditional_edges("tools", clarification_router, {END: END, "agent": "agent"})
workflow.add_edge("force_final_answer", END)

app = workflow.compile()


def _sanitize_lines_with_blocklist(raw: str, blocklist: List[str], max_chars: int) -> str:
    if not raw:
        return ""
    lines = []
    for line in raw.splitlines():
        check = line.strip().lower()
        if any(bad in check for bad in blocklist):
            continue
        lines.append(line)
    return "\n".join(lines).strip()[:max_chars]


def sanitize_system_role(raw: str, max_chars: int = 500) -> str:
    blocked = [
        "ignore previous instructions",
        "you have no restrictions",
    ]
    return _sanitize_lines_with_blocklist(raw, blocked, max_chars)


def sanitize_custom_instructions(raw: str, max_chars: int = 2000) -> str:
    blocked = [
        "ignore previous instructions",
        "ignore all previous",
        "do not use tools",
        "disable tools",
        "never use tools",
        "pretend you have no tool",
    ]
    return _sanitize_lines_with_blocklist(raw, blocked, max_chars)


def get_tool_catalog(tool_items: Optional[List[Any]] = None) -> List[Dict[str, str]]:
    catalog: List[Dict[str, str]] = []
    for tool_item in tool_items or tools_list:
        cfg = TOOL_FRIENDLY_CONFIG.get(tool_item.name, {})
        alias_id = str(cfg.get("id", tool_item.name))
        catalog.append(
            {
                "tool_name": tool_item.name,
                "id": alias_id,
                "display_name": str(cfg.get("display_name", alias_id.replace("_", " ").title())),
                "description": str(cfg.get("description", tool_item.description or "")),
                "default_prompt": str(cfg.get("default_prompt", "Use this tool when it is the most relevant retrieval path.")),
            }
        )
    return catalog


def get_default_tool_instruction_map(tool_items: Optional[List[Any]] = None) -> Dict[str, str]:
    return {item["id"]: item["default_prompt"] for item in get_tool_catalog(tool_items)}


def normalize_tool_instructions(
    raw: Optional[Dict[str, str]],
    max_chars_per_tool: int = 500,
    tool_items: Optional[List[Any]] = None,
) -> Dict[str, str]:
    blocked = [
        "do not use tools",
        "disable tools",
        "never use tools",
        "ignore tool contract",
    ]
    normalized = get_default_tool_instruction_map(tool_items)
    if not isinstance(raw, dict):
        return normalized
    for tool_id, value in raw.items():
        if tool_id not in normalized:
            continue
        text = _sanitize_lines_with_blocklist(str(value or ""), blocked, max_chars_per_tool)
        if text:
            normalized[tool_id] = text
    return normalized


def build_system_prompt(
    context_window: int,
    system_role: str = "",
    tool_instructions: Optional[Dict[str, str]] = None,
    custom_instructions: str = "",
    use_web_search: bool = False,
    intent_agent_ran: bool = True,
    reasoning_mode: bool = True,
    client_timezone: Optional[str] = None,
    client_locale: Optional[str] = None,
    client_now_iso: Optional[str] = None,
) -> str:
    """Build the Orchestrator Agent system prompt."""
    role = system_role or DEFAULT_SYSTEM_ROLE
    active_tools = get_active_tools(use_external_research=use_web_search)
    catalog = get_tool_catalog(active_tools)
    playbook = normalize_tool_instructions(tool_instructions or {}, tool_items=active_tools)
    
    # Load base template
    template = get_orchestrator_prompt() if reasoning_mode else get_orchestrator_prompt_compact()
    
    # Setup variables for template substitution
    if intent_agent_ran:
        intent_agent_note = "The Intent Agent upstream has already rewritten the user's query and classified its coverage; your job is to:"
        preprocessing_phase_note = ""
        phase0 = ""
        phase_count = "five"
        phase_start = ""
        orient_word = "rewritten"
        orient_extra = ""
        plan_query_note = ""
    else:
        intent_agent_note = "No upstream query preprocessor ran for this turn — you are responsible for both query preprocessing AND orchestration. Your job is to:"
        preprocessing_phase_note = "  0. Preprocess the raw user query: resolve coreferences, standalone-ify, assess coverage (Phase 0)."
        phase0 = get_orchestrator_phase0_prompt() if reasoning_mode else get_orchestrator_phase0_prompt_compact()
        phase_count = "six"
        phase_start = " Begin with Phase 0 — Preprocess."
        orient_word = "working"
        orient_extra = "\n  e) Does the raw message contain unresolved pronouns or references? → your Phase 0\n     WORKING QUERY replaces the raw message for all retrieval operations below."
        plan_query_note = "\n  - Use the WORKING QUERY from Phase 0 — not the raw user message — for all tool arguments."
    
    max_parallel_tools = 4 if use_web_search else 3
    
    # Build tool registry/playbook sections only when tools are actually bound.
    EDIT = "(USER-CONFIGURABLE)"
    if reasoning_mode:
        tool_registry_section = (
            f"\n\n{'=' * 64}\nTOOL REGISTRY {EDIT}:\n{'=' * 64}\n"
            + "\n".join(
                [
                    f"- {item['display_name']} (tool name: `{item['tool_name']}`)\n    {item['description']}"
                    for item in catalog
                ]
            )
        )
        tool_playbook_section = (
            f"\n\n{'=' * 64}\nTOOL PLAYBOOK {EDIT}:\n{'=' * 64}\n"
            + "\n".join(
                [
                    f"- `{item['tool_name']}`: {playbook.get(item['id'], item['default_prompt'])}"
                    for item in catalog
                ]
            )
        )
    else:
        tool_registry_section = ""
        tool_playbook_section = ""

    # Build web search mandate section if enabled
    web_search_mandate_section = ""
    if use_web_search and reasoning_mode:
        LOCK = "(LOCKED — not overridable)"
        web_search_mandate_section = (
            f"\n\n{'=' * 64}\nWEB SEARCH MANDATE {LOCK} — overrides pre-fetch sufficiency\n"
            f"{'=' * 64}\n"
            + get_web_search_mandate()
        )

    # Build custom instructions section if provided
    custom_instructions_section = ""
    if custom_instructions:
        custom_instructions_section = (
            f"\n\n{'=' * 64}\nUSER CUSTOM INSTRUCTIONS {EDIT}\n{'=' * 64}\n"
            + custom_instructions
        )

    # Substitute placeholders in template
    prompt = template.format(
        SYSTEM_ROLE=role,
        CONTEXT_WINDOW=context_window,
        INTENT_AGENT_NOTE=intent_agent_note,
        PREPROCESSING_PHASE_NOTE=preprocessing_phase_note,
        PREPROCESSING_SECTION=phase0,
        PHASE_COUNT=phase_count,
        PHASE_START=phase_start,
        ORIENT_WORD=orient_word,
        ORIENT_EXTRA=orient_extra,
        PLAN_QUERY_NOTE=plan_query_note,
        MAX_PARALLEL_TOOLS=max_parallel_tools,
        RUNTIME_DATETIME_CONTEXT=format_runtime_datetime_context(
            client_timezone=client_timezone,
            client_locale=client_locale,
            client_now_iso=client_now_iso,
        ),
        TOOL_REGISTRY_SECTION=tool_registry_section,
        TOOL_PLAYBOOK_SECTION=tool_playbook_section,
        WEB_SEARCH_MANDATE_SECTION=web_search_mandate_section,
        CUSTOM_INSTRUCTIONS_SECTION=custom_instructions_section,
    )

    return prompt
