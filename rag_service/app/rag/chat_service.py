"""
chat_service.py - Business logic for chat endpoints in RAG Service

This module provides:
- Shared retrieval prefetching for agent-pattern chat runtimes.
"""

import asyncio
import logging
import os
from typing import List, Dict, Any, cast

from app.models.llm_server_client import (
    get_embedding_model,
    DEFAULT_TOKEN_BUDGET,
    RATIO_SEMANTIC_MEMORY,
    CHARS_PER_TOKEN,
    compute_prefetch_budget,
)
from app.db import (
    get_recent_messages,
    MessageRole,
    get_thread_shape,
)
from app.rag.retrieval import (
    fetch_semantic_history,
    get_document_metadata_lookup,
    group_document_chunks,
    rerank_document_chunks,
)
from app.time_utils import maybe_iso_utc_z

logger = logging.getLogger(__name__)


async def prefetch_context(
    thread_id: str,
    raw_question: str,
    embed_model_name: str,
    context_window: int,
    use_web_search: bool,
    use_reranker: bool,
) -> Dict[str, Any]:
    """
    Gather all retrieval context in parallel BEFORE any LLM call.

    Runs four async tasks concurrently:
      1. Recent verbatim conversation turns (DB, position-based)
      2. Semantic chat-memory recall (vector search, raw question)
      3. Document evidence (vector search, raw question)
      4. Conversation stats + uploaded document list (DB metadata)

    Returns a bundle dict with text strings ready for injection into LLM
    system prompts, plus structured metadata (document_sources, used_chat_ids,
    document list) for state initialization.

    Design principles:
    - Parallelism eliminates the latency penalty vs sequential DB + vector calls.
    - Budget computation is LLM-agnostic: ratios scale proportionally with any
      context window (4 K → 1 M tokens).
    - Both agents receive the same bundle; no re-fetching between Intent and
      Orchestrator for the same raw question.
    """
    from app.db.vector import get_vector_db
    from app.models.retry import invoke_with_retry

    budget = compute_prefetch_budget(context_window)

    # Embed the raw question ONCE and share the vector across parallel tasks
    embed_model = get_embedding_model(embed_model_name)
    shared_query_vector = await invoke_with_retry(embed_model.aembed_query, raw_question)

    async def _fetch_recent() -> str:
        msgs = await get_recent_messages(thread_id, limit=budget["recent_turn_limit"] * 2)
        lines: List[str] = []
        used_chars = 0
        budget_chars = budget["recent_history_chars"]
        for msg in reversed(msgs):
            role = "User" if msg.role == MessageRole.USER.value else "Assistant"
            text = (msg.context_compact or msg.content or "").strip()
            if not text:
                continue
            message_created_at = maybe_iso_utc_z(msg.created_at)
            entry = f"{role} at {message_created_at}: {text}" if message_created_at else f"{role}: {text}"
            if used_chars + len(entry) > budget_chars:
                break
            lines.append(entry)
            used_chars += len(entry)
        lines.reverse()
        return "\n\n".join(lines)

    async def _fetch_stats_and_docs() -> Dict[str, Any]:
        """
        Read the pre-maintained thread shape.
        O(1) lookup — no message scanning needed.
        """
        shape = await get_thread_shape(thread_id)
        stats = {
            "total_messages": shape["total_qa_pairs"] * 2,  # user + assistant
            "estimated_history_tokens": round(shape["total_qa_chars"] / 4, 0),
        }
        # Build indexed document list (exclude pending/failed if no chunks yet)
        documents = [
            {
                "index": i + 1,
                "file_name": meta["file_name"],
                "file_hash": fh,
                "source_type": meta.get("source_type"),
                "document_available_in_thread_at": meta.get("document_available_in_thread_at"),
                "chunk_count": meta.get("chunk_count"),
                "total_chars": meta.get("total_chars"),
                "word_count": meta.get("word_count"),
                "page_count": meta.get("page_count"),
                "sentence_count": meta.get("sentence_count"),
                "languages": meta.get("languages"),
                "filetype": meta.get("filetype"),
            }
            for i, (fh, meta) in enumerate(shape["documents"].items())
        ]
        return {"stats": stats, "documents": documents}

    async def _fetch_semantic() -> tuple:
        try:
            return await fetch_semantic_history(
                thread_id=thread_id,
                query_vector=shared_query_vector,
                query_text=raw_question,
                limit=budget["semantic_limit"],
                char_budget=budget["semantic_history_chars"],
                use_reranker=use_reranker,
                embedding_model_name=embed_model_name,
            )
        except Exception as exc:
            logger.warning(f"Prefetch semantic history failed: {exc}")
            return "", []

    async def _fetch_documents() -> tuple:
        try:
            db = get_vector_db()
            limit = budget["document_limit"]
            rerank_fetch_k = limit
            shape = await get_thread_shape(thread_id)
            thread_file_hashes = list(shape.get("documents", {}).keys())
            if not thread_file_hashes:
                return "", []
            raw_chunks = await db.search_knowledge_sources(
                thread_id=thread_id,
                query_vector=shared_query_vector,
                embedding_model_name=embed_model_name,
                limit=rerank_fetch_k,
                file_hashes=thread_file_hashes,
                query_text=raw_question,
            )
            if not raw_chunks:
                logger.error(
                    "Missing document vectors for thread %s (files=%d, embed_model=%s). "
                    "Recovery is only triggered on thread open.",
                    thread_id,
                    len(thread_file_hashes),
                    embed_model_name,
                )
                return "", []
            document_lookup = await get_document_metadata_lookup(thread_id)
            if use_reranker:
                raw_chunks = await rerank_document_chunks(raw_question, raw_chunks)
            return group_document_chunks(
                raw_chunks,
                document_lookup,
                char_budget=budget["document_context_chars"],
            )
        except Exception as exc:
            logger.warning(f"Prefetch document evidence failed: {exc}")
            return "", []

    # Run all fetches in parallel
    results = await asyncio.gather(
        _fetch_recent(),
        _fetch_stats_and_docs(),
        _fetch_semantic(),
        _fetch_documents(),
        return_exceptions=True,
    )

    recent_text = cast(str, results[0]) if not isinstance(results[0], Exception) else ""
    meta = cast(Dict[str, Any], results[1]) if not isinstance(results[1], Exception) else {"stats": {}, "documents": []}
    semantic_result = results[2] if not isinstance(results[2], Exception) else ("", [])
    document_result = results[3] if not isinstance(results[3], Exception) else ("", [])
    web_result = ("", [])

    semantic_text, used_chat_ids = semantic_result if isinstance(semantic_result, tuple) else ("", [])
    document_text, document_sources = document_result if isinstance(document_result, tuple) else ("", [])
    web_text, web_sources = web_result if isinstance(web_result, tuple) else ("", [])

    return {
        "recent_history_text":   recent_text,
        "semantic_history_text": semantic_text,
        "document_evidence_text":     document_text,
        "web_evidence_text":     web_text,
        "stats":                 meta.get("stats", {}),
        "documents":             meta.get("documents", []),
        "document_sources":      document_sources,
        "web_sources":           web_sources,
        "used_chat_ids":         used_chat_ids,
        "budget":                budget,
    }
