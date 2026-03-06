"""
chat_service.py - Business logic for chat endpoints in RAG Service

This module provides:
- Thread-based chat with semantic memory (handle_thread_chat)
"""

import asyncio
import logging
import time
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
    INTENT_AGENT_MAX_ITERATIONS,
    MAX_ITERATIONS_SUFFICIENT_COVERAGE,
    MAX_ITERATIONS_PROBABLY_SUFFICIENT_COVERAGE,
    WEB_SEARCH_ITERATION_BONUS,
)
from database import (
    create_message,
    get_recent_messages,
    update_message_context_compact,
    MessageRole,
    get_thread_shape,
    increment_qa_stats,
)
from reasoning import normalize_ai_response
from retrieval import fetch_semantic_history, get_document_name_lookup, group_pdf_chunks
from web_prefetch import prefetch_web_context

logger = logging.getLogger(__name__)


async def prefetch_context(
    thread_id: str,
    raw_question: str,
    embed_model_name: str,
    context_window: int,
    use_web_search: bool,
    reasoning_mode: bool,
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
    from vectordb.qdrant import get_qdrant
    from agent import invoke_with_retry

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
        """
        Read thread shape from the pre-maintained thread_stats table.
        O(1) lookup — no message scanning needed.
        """
        shape = await get_thread_shape(thread_id)
        stats = {
            "total_messages": shape["total_qa_pairs"] * 2,  # user + assistant
            "estimated_history_tokens": round(shape["total_qa_chars"] / 4, 0),
        }
        # Build indexed document list (exclude pending/failed if no chunks yet)
        documents = [
            {"index": i + 1, "file_name": meta["file_name"], "file_hash": fh}
            for i, (fh, meta) in enumerate(shape["documents"].items())
        ]
        return {"stats": stats, "documents": documents}

    async def _fetch_semantic() -> tuple:
        try:
            return await fetch_semantic_history(
                thread_id=thread_id,
                query_vector=shared_query_vector,
                limit=budget["semantic_limit"],
                char_budget=budget["semantic_history_chars"],
            )
        except Exception as exc:
            logger.warning(f"Prefetch semantic history failed: {exc}")
            return "", []

    async def _fetch_pdf() -> tuple:
        try:
            db = get_qdrant()
            raw_chunks = await db.search_knowledge_sources(
                thread_id=thread_id,
                query_vector=shared_query_vector,
                limit=budget["pdf_limit"],
            )

            hash_to_name = await get_document_name_lookup(thread_id)
            return group_pdf_chunks(
                raw_chunks,
                hash_to_name,
                char_budget=budget["pdf_context_chars"],
            )
        except Exception as exc:
            logger.warning(f"Prefetch PDF evidence failed: {exc}")
            return "", []

    async def _fetch_web() -> tuple:
        return await prefetch_web_context(
            raw_question=raw_question,
            thread_id=thread_id,
            embed_model_name=embed_model_name,
            use_web_search=use_web_search,
            reasoning_mode=reasoning_mode,
            logger=logger,
        )

    # Run all fetches in parallel
    results = await asyncio.gather(
        _fetch_recent(),
        _fetch_stats_and_docs(),
        _fetch_semantic(),
        _fetch_pdf(),
        _fetch_web(),
        return_exceptions=True,
    )

    recent_text: str = results[0] if not isinstance(results[0], Exception) else ""
    meta: Dict[str, Any] = results[1] if not isinstance(results[1], Exception) else {"stats": {}, "documents": []}
    semantic_result = results[2] if not isinstance(results[2], Exception) else ("", [])
    pdf_result = results[3] if not isinstance(results[3], Exception) else ("", [])
    web_result = results[4] if not isinstance(results[4], Exception) else ("", [])

    semantic_text, used_chat_ids = semantic_result if isinstance(semantic_result, tuple) else ("", [])
    pdf_text, pdf_sources = pdf_result if isinstance(pdf_result, tuple) else ("", [])
    web_text, web_sources = web_result if isinstance(web_result, tuple) else ("", [])

    return {
        "recent_history_text":   recent_text,
        "semantic_history_text": semantic_text,
        "pdf_evidence_text":     pdf_text,
        "web_evidence_text":     web_text,
        "stats":                 meta.get("stats", {}),
        "documents":             meta.get("documents", []),
        "pdf_sources":           pdf_sources,
        "web_sources":           web_sources,
        "used_chat_ids":         used_chat_ids,
        "budget":                budget,
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
    use_intent_agent = getattr(req, 'use_intent_agent', True)
    reasoning_mode = getattr(req, 'reasoning_mode', True)
    if use_intent_agent is None:
        use_intent_agent = True
    # NOTE: intent_agent_max_iterations is passed to the LangGraph Intent Agent State. 
    # It is currently not exposed in the UI (hardcoded to single-pass logic); 
    # it can be re-exposed when the Intent Agent is upgraded with tools or multi-step reasoning.
    intent_agent_max_iterations = getattr(req, 'intent_agent_max_iterations', None) or INTENT_AGENT_MAX_ITERATIONS
    
    start_total = time.perf_counter()
    intent_duration = 0.0
    orchestrator_duration = 0.0
    
    try:
        from agent import app as agent_app, intent_app, AgentState
        
        # 1. Run context pre-fetch (no LLM cost)
        prefetch_bundle = await prefetch_context(
            thread_id=thread_id,
            raw_question=question,
            embed_model_name=embed_model,
            context_window=context_window,
            use_web_search=use_web_search,
            reasoning_mode=reasoning_mode,
        )

        # 2. Analyze intent using the Intent Agent (optional)
        intent = {
            "status": "CLEAR_STANDALONE",
            "rewritten_query": question,
            "clarification_options": None,
            "context_coverage": "INSUFFICIENT",
        }

        if use_intent_agent:
            intent_state = {
                "messages": [HumanMessage(content=question)],
                "thread_id": thread_id,
                "llm_model": llm_model,
                "context_window": context_window,
                "iteration_count": 0,
                "max_iterations": intent_agent_max_iterations,
                "intent_result": None,
                "pre_fetch_bundle": prefetch_bundle,
                "reasoning_mode": reasoning_mode,
            }
            
            logger.info(f"Invoking Intent Agent for thread {thread_id}")
            intent_start = time.perf_counter()
            intent_result_state = await intent_app.ainvoke(intent_state, config={"configurable": {"thread_id": thread_id}})
            intent_duration = time.perf_counter() - intent_start
            
            if intent_result_state.get("intent_result"):
                intent = intent_result_state["intent_result"]
        else:
            logger.info(f"Intent Agent disabled for thread {thread_id}, skipping")
        
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
         
        # Normalize question from intent results
        question = intent.get("rewritten_query") or question
        reference_type = intent.get("reference_type", "NONE")

        logger.info(
            f"Intent analysis done for thread {thread_id} | "
            f"rewritten_query: {question} | "
            f"reference_type: {intent.get('reference_type', 'NONE')} | "
            f"context_coverage: {intent.get('context_coverage', 'PROBABLY_SUFFICIENT')}"
        )

        # Cap orchestrator iterations based on intent's coverage signal
        # SUFFICIENT          → 2 rounds max  (pre-fetch + 1 targeted tool if needed)
        # PROBABLY_SUFFICIENT → 4 rounds max
        # INSUFFICIENT        → full budget (default max_iterations)
        # When web search is enabled, add 2 extra iterations so the agent has room to
        # call search_web after an initial search_documents call that may return nothing.
        coverage = intent.get("context_coverage", "PROBABLY_SUFFICIENT")
        web_bonus = WEB_SEARCH_ITERATION_BONUS if use_web_search else 0
        if coverage == "SUFFICIENT":
            effective_max_iterations = min(max_iterations, MAX_ITERATIONS_SUFFICIENT_COVERAGE + web_bonus)
        elif coverage == "PROBABLY_SUFFICIENT":
            effective_max_iterations = min(max_iterations, MAX_ITERATIONS_PROBABLY_SUFFICIENT_COVERAGE + web_bonus)
        else:
            effective_max_iterations = max_iterations

        initial_state = {
            "messages": [HumanMessage(content=question)],
            "thread_id": thread_id,
            "llm_model": llm_model,
            "embedding_model": embed_model,
            "context_window": context_window,
            "use_web_search": use_web_search,
            # Pre-seed sources from the prefetch pass; tool calls will extend these lists
            "pdf_sources": list(prefetch_bundle.get("pdf_sources", [])),
            "web_sources": list(prefetch_bundle.get("web_sources", [])),
            "used_chat_ids": list(prefetch_bundle.get("used_chat_ids", [])),
            "clarification_options": None,
            "iteration_count": 0,
            "max_iterations": effective_max_iterations,
            "system_role": system_role,
            "tool_instructions": tool_instructions,
            "custom_instructions": custom_instructions,
            "pre_fetch_bundle": prefetch_bundle,
            # Signal whether the Intent Agent ran; Orchestrator adapts its prompting strategy
            "intent_agent_ran": use_intent_agent,
            "reasoning_mode": reasoning_mode,
            "working_query": question,
            "intent_reference_type": reference_type,
        }
        
        config = {
            "configurable": {
                "thread_id": thread_id,
                "embedding_model": embed_model,
                "context_window": context_window,
                "use_web_search": use_web_search,
            }
        }
        
        logger.info(f"Invoking Orchestrator Agent for thread {thread_id}")
        orchestrator_start = time.perf_counter()
        result = await agent_app.ainvoke(initial_state, config=config)
        orchestrator_duration = time.perf_counter() - orchestrator_start
        total_duration = time.perf_counter() - start_total
        
        logger.info(
            f"CHAT COMPLETED [thread {thread_id}] | "
            f"Intent: {intent_duration:.2f}s | "
            f"Orchestrator: {orchestrator_duration:.2f}s | "
            f"Total: {total_duration:.2f}s | "
            f"LLM: {llm_model} | "
            f"Agent_Iterations: {result.get('iteration_count', 0)}"
        )
        
        final_messages = result.get("messages", [])
        normalized = normalize_ai_response(final_messages[-1] if final_messages else None)
        if not normalized["answer"]:
            last_msg = final_messages[-1] if final_messages else None
            logger.warning(
                f"Empty answer from agent for thread {thread_id}. "
                f"Last message type={type(last_msg).__name__}, "
                f"content={repr(getattr(last_msg, 'content', None))!r}, "
                f"tool_calls={getattr(last_msg, 'tool_calls', None)}"
            )
        answer = normalized["answer"] or "I was unable to compose an answer. Please try rephrasing your question."
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

        # Update thread stats: increment QA pair counter
        try:
            qa_chars = len(req.question) + len(answer)
            await increment_qa_stats(thread_id, qa_chars)
        except Exception as stats_err:
            logger.warning(f"thread_stats QA increment skipped: {stats_err}")

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
