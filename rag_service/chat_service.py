"""
chat_service.py - Business logic for chat endpoints in RAG Service

This module provides:
- Legacy chat handling (handle_chat)
- Thread-based chat with semantic memory (handle_thread_chat)
"""

import asyncio
import logging
from typing import List, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage

from rag import index_chat_memory_for_thread
from models import (
    get_embedding_model,
    DEFAULT_TOKEN_BUDGET,
    DEFAULT_MAX_ITERATIONS,
    RATIO_SEMANTIC_MEMORY,
    CHARS_PER_TOKEN,
    compute_prefetch_budget,
)
from database import (
    create_message,
    get_recent_messages,
    get_thread_files,
    get_thread_messages,
    update_message_context_compact,
    MessageRole,
)
from reasoning import normalize_ai_response

logger = logging.getLogger(__name__)


async def prefetch_context(
    thread_id: str,
    raw_question: str,
    embed_model_name: str,
    context_window: int,
) -> Dict[str, Any]:
    """
    Gather all retrieval context in parallel BEFORE any LLM call.

    Runs four async tasks concurrently:
      1. Recent verbatim conversation turns (DB, position-based)
      2. Semantic chat-memory recall (vector search, raw question)
      3. PDF document evidence (vector search, raw question)
      4. Conversation stats + uploaded document list (DB metadata)

    Returns a bundle dict with text strings ready for injection into LLM
    system prompts, plus structured metadata (pdf_sources, used_chat_ids,
    document list) for state initialization.

    Design principles:
    - Parallelism eliminates the latency penalty vs sequential DB + vector calls.
    - Budget computation is LLM-agnostic: ratios scale proportionally with any
      context window (4 K → 1 M tokens).
    - Both agents receive the same bundle; no re-fetching between Intent and
      Orchestrator for the same raw question.
    """
    from vectordb.qdrant import QdrantAdapter
    from agent import invoke_with_retry

    budget = compute_prefetch_budget(context_window)

    async def _fetch_recent() -> str:
        msgs = await get_recent_messages(thread_id, limit=budget["recent_turn_limit"] * 2)
        lines: List[str] = []
        used_chars = 0
        budget_chars = budget["recent_history_chars"]
        for msg in reversed(msgs):
            role = "User" if msg.role == MessageRole.USER else "Assistant"
            text = (msg.context_compact or msg.content or "").strip()
            if not text:
                continue
            entry = f"{role}: {text}"
            if used_chars + len(entry) > budget_chars:
                break
            lines.append(entry)
            used_chars += len(entry)
        lines.reverse()
        return "\n\n".join(lines)

    async def _fetch_stats_and_docs() -> Dict[str, Any]:
        import asyncio as _asyncio
        msgs, files = await _asyncio.gather(
            get_thread_messages(thread_id, limit=2000),
            get_thread_files(thread_id),
        )
        total = len(msgs)
        avg_len = sum(len(m.content) for m in msgs) / total if total else 0
        stats = {
            "total_messages": total,
            "estimated_history_tokens": round((avg_len * total) / 4, 0),
        }
        documents = [
            {"index": i + 1, "file_name": f.file_name, "file_hash": f.file_hash}
            for i, f in enumerate(files)
        ]
        return {"stats": stats, "documents": documents}

    async def _fetch_semantic() -> tuple:
        try:
            embed_model = get_embedding_model(embed_model_name)
            query_vector = await invoke_with_retry(embed_model.aembed_query, raw_question)
            db = QdrantAdapter()
            recalled = await db.search_chat_memory(
                thread_id=thread_id,
                query_vector=query_vector,
                limit=budget["semantic_limit"],
            )
            used_ids = [m["message_id"] for m in recalled if m.get("message_id")]
            parts: List[str] = []
            used_chars = 0
            for mem in recalled:
                text = mem.get("text", "")
                if used_chars + len(text) > budget["semantic_history_chars"]:
                    break
                parts.append(text)
                used_chars += len(text)
            return "\n\n---\n\n".join(parts), used_ids
        except Exception as exc:
            logger.warning(f"Prefetch semantic history failed: {exc}")
            return "", []

    async def _fetch_pdf() -> tuple:
        try:
            embed_model = get_embedding_model(embed_model_name)
            query_vector = await invoke_with_retry(embed_model.aembed_query, raw_question)
            db = QdrantAdapter()
            raw_chunks = await db.search_pdf_chunks(
                thread_id=thread_id,
                query_vector=query_vector,
                limit=budget["pdf_limit"],
            )
            sources: List[Dict[str, Any]] = []
            parts: List[str] = []
            used_chars = 0
            for chunk in raw_chunks:
                text = chunk.get("text", "")
                if used_chars + len(text) > budget["pdf_context_chars"]:
                    break
                parts.append(text)
                used_chars += len(text)
                sources.append({
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "file_hash": chunk.get("file_hash"),
                    "score": chunk.get("score", 0.0),
                })
            return "\n\n".join(parts), sources
        except Exception as exc:
            logger.warning(f"Prefetch PDF evidence failed: {exc}")
            return "", []

    # Run all four fetches in parallel
    results = await asyncio.gather(
        _fetch_recent(),
        _fetch_stats_and_docs(),
        _fetch_semantic(),
        _fetch_pdf(),
        return_exceptions=True,
    )

    recent_text: str = results[0] if not isinstance(results[0], Exception) else ""
    meta: Dict[str, Any] = results[1] if not isinstance(results[1], Exception) else {"stats": {}, "documents": []}
    semantic_result = results[2] if not isinstance(results[2], Exception) else ("", [])
    pdf_result = results[3] if not isinstance(results[3], Exception) else ("", [])

    semantic_text, used_chat_ids = semantic_result if isinstance(semantic_result, tuple) else ("", [])
    pdf_text, pdf_sources = pdf_result if isinstance(pdf_result, tuple) else ("", [])

    return {
        "recent_history_text":   recent_text,
        "semantic_history_text": semantic_text,
        "pdf_evidence_text":     pdf_text,
        "stats":                 meta.get("stats", {}),
        "documents":             meta.get("documents", []),
        "pdf_sources":           pdf_sources,
        "used_chat_ids":         used_chat_ids,
        "budget":                budget,
    }


async def _build_recent_history_messages(
    thread_id: str,
    context_window: int,
) -> List[Any]:
    """
    Load a compact recent transcript so follow-up questions remain grounded
    even when the model skips history-related tool calls.
    """
    safe_window = max(DEFAULT_TOKEN_BUDGET, int(context_window or DEFAULT_TOKEN_BUDGET))
    # Reuse the semantic-memory allocation ratio as the direct recent-history budget.
    history_budget_chars = int(safe_window * RATIO_SEMANTIC_MEMORY * CHARS_PER_TOKEN)

    # Fetch a ratio-scaled candidate pool, then pack newest->oldest within budget.
    candidate_limit = max(1, int((safe_window * RATIO_SEMANTIC_MEMORY) / max(1, DEFAULT_MAX_ITERATIONS)))
    recent = await get_recent_messages(thread_id, limit=candidate_limit)

    packed_reversed: List[Any] = []
    used_chars = 0

    for msg in reversed(recent):
        if msg.role == MessageRole.USER:
            text = (msg.context_compact or msg.content or "").strip()
            message_obj = HumanMessage
        elif msg.role == MessageRole.ASSISTANT:
            text = (msg.context_compact or msg.content or "").strip()
            message_obj = AIMessage
        else:
            continue

        if not text:
            continue

        text_chars = len(text)
        if packed_reversed and used_chars + text_chars > history_budget_chars:
            break

        packed_reversed.append(message_obj(content=text))
        used_chars += text_chars

    return list(reversed(packed_reversed))


async def handle_chat(req) -> Dict[str, Any]:
    """
    Legacy chat handler for backward compatibility.
    Uses collection-based retrieval without semantic memory.
    """
    from agent import app as agent_app
    chat_history = []
    for msg in req.history:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))

    inputs = {
        "messages": chat_history + [HumanMessage(content=req.question)],
        "question": req.question,
        "llm_model": req.llm_model,
        "embedding_model": req.embedding_model,
        "collection_name": req.collection_name,
        "use_web_search": req.use_web_search,
        "context": "",
        "web_context": "",
        "answer": "",
        "iteration_count": 0,
        "max_iterations": getattr(req, 'max_iterations', DEFAULT_MAX_ITERATIONS),
        "system_role": "",
        "tool_instructions": {},
        "custom_instructions": "",
    }
    
    # Provide a dummy config for backward compatibility
    config = {
        "configurable": {
            "thread_id": "legacy_thread",
            "embedding_model": req.embedding_model,
            "context_window": 8000
        }
    }
    
    result = await agent_app.ainvoke(inputs, config=config)
    messages = result.get("messages", [])
    normalized = normalize_ai_response(messages[-1] if messages else None)
    answer = normalized["answer"] or "Error"
    return {
        "answer": answer,
        "reasoning": normalized["reasoning"],
        "reasoning_available": normalized["reasoning_available"],
        "reasoning_format": normalized["reasoning_format"],
        "context": "Legacy context retrieval unsupported",
    }


async def handle_thread_chat(
    thread_id: str,
    req,  # ThreadChatRequest
    embed_model: str
) -> Dict[str, Any]:
    """
    Thread-based chat using Orchestrator Agent with dynamic memory, PDF search, and web tools.
    """
    question = req.question
    llm_model = req.llm_model
    use_web_search = getattr(req, 'use_web_search', False)
    context_window = getattr(req, 'context_window', DEFAULT_TOKEN_BUDGET)
    max_iterations = getattr(req, 'max_iterations', None) or DEFAULT_MAX_ITERATIONS
    system_role = getattr(req, 'system_role_override', "") or ""
    tool_instructions = getattr(req, 'tool_instructions_override', None) or {}
    custom_instructions = getattr(req, 'custom_instructions_override', "") or ""
    
    try:
        from agent import app as agent_app, intent_app, AgentState
        
        # 1. Run recent-history build and context pre-fetch in parallel (no LLM cost)
        recent_history_messages, prefetch_bundle = await asyncio.gather(
            _build_recent_history_messages(
                thread_id=thread_id,
                context_window=context_window,
            ),
            prefetch_context(
                thread_id=thread_id,
                raw_question=question,
                embed_model_name=embed_model,
                context_window=context_window,
            ),
        )

        # 2. Analyze intent using the Intent Agent
        intent_state = {
            "messages": recent_history_messages + [HumanMessage(content=question)],
            "thread_id": thread_id,
            "llm_model": llm_model,
            "context_window": context_window,
            "iteration_count": 0,
            "max_iterations": 3,
            "intent_result": None,
            "pre_fetch_bundle": prefetch_bundle,
        }
        
        intent_config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
        
        logger.info(f"Invoking Intent Agent for thread {thread_id}")
        intent_result_state = await intent_app.ainvoke(intent_state, config=intent_config)
        intent = intent_result_state.get("intent_result") or {
            "status": "CLEAR_STANDALONE", 
            "rewritten_query": question, 
            "clarification_options": None
        }
        
        # If ambiguous, return early with clarification options
        if intent["status"] == "AMBIGUOUS" and intent.get("clarification_options"):
            return {
                "answer": "I'm not sure I understand. Could you clarify which of these you meant?",
                "clarification_options": intent["clarification_options"],
                "rewritten_query": intent.get("rewritten_query") or question,
                "user_message_id": None,
                "assistant_message_id": None,
                "used_chat_ids": [],
                "pdf_sources": [],
                "web_sources": [],
                "reasoning": "",
                "reasoning_available": False,
                "reasoning_format": "none",
                "context": "Needs human-in-the-loop clarification."
            }
         
        logger.info(
            f"Intent analysis done for thread {thread_id} | "
            f"rewritten_query: {intent['rewritten_query']} | "
            f"reference_type: {intent.get('reference_type', 'NONE')} | "
            f"context_coverage: {intent.get('context_coverage', 'PROBABLY_SUFFICIENT')}"
        )
        # Use rewritten query for the rest of the path
        if intent.get("rewritten_query"):
            question = intent["rewritten_query"]

        # Cap orchestrator iterations based on intent's coverage signal:
        # SUFFICIENT          → 2 rounds max  (pre-fetch + 1 targeted tool if needed)
        # PROBABLY_SUFFICIENT → 4 rounds max
        # INSUFFICIENT        → full budget (default max_iterations)
        coverage = intent.get("context_coverage", "PROBABLY_SUFFICIENT")
        if coverage == "SUFFICIENT":
            effective_max_iterations = min(max_iterations, 2)
        elif coverage == "PROBABLY_SUFFICIENT":
            effective_max_iterations = min(max_iterations, 4)
        else:
            effective_max_iterations = max_iterations

        initial_state = {
            "messages": recent_history_messages + [HumanMessage(content=question)],
            "thread_id": thread_id,
            "llm_model": llm_model,
            "embedding_model": embed_model,
            "context_window": context_window,
            "use_web_search": use_web_search,
            # Pre-seed sources from the prefetch pass; tool calls will extend these lists
            "pdf_sources": list(prefetch_bundle.get("pdf_sources", [])),
            "web_sources": [],
            "used_chat_ids": list(prefetch_bundle.get("used_chat_ids", [])),
            "clarification_options": None,
            "iteration_count": 0,
            "max_iterations": effective_max_iterations,
            "system_role": system_role,
            "tool_instructions": tool_instructions,
            "custom_instructions": custom_instructions,
            "pre_fetch_bundle": prefetch_bundle,
        }
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "embedding_model": embed_model,
                "context_window": context_window
            }
        }
        
        logger.info(f"Invoking Orchestrator Agent for thread {thread_id}")
        result = await agent_app.ainvoke(initial_state, config=config)
        
        final_messages = result.get("messages", [])
        normalized = normalize_ai_response(final_messages[-1] if final_messages else None)
        answer = normalized["answer"] or "Error processing request."
        pdf_sources = result.get("pdf_sources", [])
        web_sources = result.get("web_sources", [])
        used_chat_ids = result.get("used_chat_ids", [])
        clarification_options = result.get("clarification_options", None)
        
        if clarification_options:
            answer = f"I need a bit more clarification. Did you mean:\n" + "\n".join([f"- {opt}" for opt in clarification_options])
            normalized = {
                "reasoning": "",
                "reasoning_available": False,
                "reasoning_format": "none",
            }
        
        # Store messages in database
        user_message = await create_message(
            thread_id=thread_id,
            role=MessageRole.USER,
            content=req.question,
            context_compact=question if question != req.question else None
        )
        
        assistant_message = await create_message(
            thread_id=thread_id,
            role=MessageRole.ASSISTANT,
            content=answer,
            reasoning=normalized["reasoning"],
            reasoning_available=normalized["reasoning_available"],
            reasoning_format=normalized["reasoning_format"],
            web_sources=web_sources if web_sources else None,
        )
        
        # Index in semantic memory if not a clarification
        if not clarification_options:
            indexing_result = await index_chat_memory_for_thread(
                thread_id=thread_id,
                message_id=assistant_message.id,
                question=question,
                answer=answer,
                embedding_model_name=embed_model,
                llm_name=llm_model,
                context_window=context_window
            )
            compact_text = indexing_result.get("memory_compact_text") if isinstance(indexing_result, dict) else None
            if compact_text:
                await update_message_context_compact(assistant_message.id, compact_text)
        
        return {
            "answer": answer,
            "rewritten_query": question, # Return rewritten version for UI
            "user_message_id": user_message.id,
            "assistant_message_id": assistant_message.id,
            "used_chat_ids": used_chat_ids,
            "pdf_sources": pdf_sources,
            "web_sources": web_sources,
            "clarification_options": clarification_options,
            "reasoning": normalized["reasoning"],
            "reasoning_available": normalized["reasoning_available"],
            "reasoning_format": normalized["reasoning_format"],
            "context": "Context retrieved dynamically by LangGraph Orchestrator tool calls."
        }
        
    except Exception as e:
        logger.error("Error in handle_thread_chat", exc_info=True)
        raise e
