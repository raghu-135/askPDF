from __future__ import annotations

import time
import logging
from typing import Any, Dict

from app.agent_patterns.graph import TemplateCompiler
from app.db import (
    create_chat_turn,
    increment_qa_stats,
    update_message_context_compact,
)
from app.rag.indexer import index_chat_memory_for_thread
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET


logger = logging.getLogger(__name__)


async def _invoke_graph_with_partial_state(app: Any, state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    latest_state = dict(state)
    async for chunk in app.astream(state, config=config, stream_mode="values"):
        if isinstance(chunk, dict):
            latest_state = chunk
    return latest_state


async def handle_router_rag_chat(
    thread_id: str,
    req: Any,
    embed_model: str,
    *,
    resolved_spec: Dict[str, Any],
    agent_run_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute the compiled Router RAG v2 graph and persist a chat turn."""

    agent_run_id = agent_run_context.get("agent_run_id")
    question = req.question
    llm_model = req.llm_model
    use_web_search = bool(getattr(req, "use_web_search", False))
    use_reranker = getattr(req, "use_reranker", None)
    if use_reranker is None:
        use_reranker = True
    context_window = getattr(req, "context_window", None) or DEFAULT_TOKEN_BUDGET
    system_role = getattr(req, "system_role_override", "") or ""
    tool_instructions = getattr(req, "tool_instructions_override", None) or {}
    custom_instructions = getattr(req, "custom_instructions_override", "") or ""
    pattern_config = resolved_spec.get("config") if isinstance(resolved_spec.get("config"), dict) else {}
    allowed_tool_ids = pattern_config.get("allowed_tool_ids")
    allowed_tool_ids = allowed_tool_ids if isinstance(allowed_tool_ids, list) else []

    started = time.perf_counter()
    app = TemplateCompiler().compile(resolved_spec)
    telemetry_sink: Dict[str, Any] = {"node_events": [], "tool_events": []}
    config = {
        "configurable": {
            "thread_id": thread_id,
            "embedding_model": embed_model,
            "context_window": context_window,
            "use_web_search": use_web_search,
            "use_reranker": use_reranker,
            "telemetry_sink": telemetry_sink,
        }
    }
    state = {
        "agent_run_id": agent_run_id,
        "thread_id": thread_id,
        "question": question,
        "llm_model": llm_model,
        "embedding_model": embed_model,
        "context_window": context_window,
        "use_web_search": use_web_search,
        "use_reranker": use_reranker,
        "system_role": system_role,
        "tool_instructions": tool_instructions,
        "custom_instructions": custom_instructions,
        "allowed_tool_ids": allowed_tool_ids,
        "client_timezone": getattr(req, "client_timezone", None),
        "client_locale": getattr(req, "client_locale", None),
        "client_now_iso": getattr(req, "client_now_iso", None),
        "document_sources": [],
        "web_sources": [],
        "used_chat_ids": [],
        "node_events": [],
        "tool_events": [],
        "errors": [],
    }

    try:
        logger.info(
            "Router RAG run started | run_id=%s thread_id=%s pattern=%s version=%s question_chars=%s",
            agent_run_id,
            thread_id,
            agent_run_context.get("agent_pattern_id"),
            agent_run_context.get("agent_pattern_version"),
            len(question or ""),
        )
        result = await _invoke_graph_with_partial_state(app, state, config)
        answer = result.get("final_answer") or "I was unable to compose an answer. Please try rephrasing your question."
        clarification_options = result.get("clarification_options")
        status = "clarification" if clarification_options else "completed"
        metadata = {
            **agent_run_context,
            "agent_route": result.get("route"),
            "agent_route_reason": result.get("route_reason"),
            "agent_node_events": result.get("node_events", []),
            "agent_tool_events": result.get("tool_events", []),
        }
        turn = await create_chat_turn(
            thread_id=thread_id,
            question=req.question,
            answer=answer,
            rewritten_question=None,
            status=status,
            reasoning=result.get("reasoning") or "",
            reasoning_available=bool(result.get("reasoning_available")),
            reasoning_format=result.get("reasoning_format") or "none",
            web_sources=result.get("web_sources") or [],
            document_sources=result.get("document_sources") or [],
            used_chat_ids=result.get("used_chat_ids") or [],
            clarification_options=clarification_options,
            metadata=metadata,
        )
        user_message_id = f"{turn.id}:user"
        assistant_message_id = f"{turn.id}:assistant"

        if not clarification_options:
            indexing_result = await index_chat_memory_for_thread(
                thread_id=thread_id,
                message_id=turn.id,
                question=question,
                answer=answer,
                embedding_model_name=embed_model,
                llm_name=llm_model,
                context_window=context_window,
                message_created_at=turn.completed_at or turn.created_at,
            )
            compact_text = indexing_result.get("memory_compact_text") if isinstance(indexing_result, dict) else None
            if compact_text:
                await update_message_context_compact(turn.id, compact_text)

        try:
            await increment_qa_stats(thread_id, len(req.question) + len(answer))
        except Exception:
            pass

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Router RAG run completed | run_id=%s thread_id=%s route=%s status=%s elapsed_ms=%.1f document_sources=%s web_sources=%s used_chat_ids=%s node_events=%s tool_events=%s",
            agent_run_id,
            thread_id,
            result.get("route"),
            status,
            duration_ms,
            len(result.get("document_sources") or []),
            len(result.get("web_sources") or []),
            len(result.get("used_chat_ids") or []),
            len(result.get("node_events") or []),
            len(result.get("tool_events") or []),
        )

        return {
            "answer": answer,
            "rewritten_query": question,
            "chat_turn_id": turn.id,
            "user_message_id": user_message_id,
            "assistant_message_id": assistant_message_id,
            "used_chat_ids": result.get("used_chat_ids") or [],
            "document_sources": result.get("document_sources") or [],
            "web_sources": result.get("web_sources") or [],
            "clarification_options": clarification_options,
            "reasoning": result.get("reasoning") or "",
            "reasoning_available": bool(result.get("reasoning_available")),
            "reasoning_format": result.get("reasoning_format") or "none",
            "context": "Context retrieved by compiled Router RAG Agent pattern.",
            "route": result.get("route"),
            "route_reason": result.get("route_reason"),
            "node_events": result.get("node_events") or [],
            "tool_events": result.get("tool_events") or [],
            "duration_ms": duration_ms,
            **agent_run_context,
        }
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        partial_result = result if isinstance(locals().get("result"), dict) else state
        logger.exception(
            "Router RAG run failed | run_id=%s thread_id=%s elapsed_ms=%.1f",
            agent_run_id,
            thread_id,
            duration_ms,
        )
        fallback_answer = (
            "I'm sorry, I encountered a technical error while processing your request. "
            "Please try again in a moment or try rephrasing your question."
        )
        error_payload = {
            "code": "router_rag_execution_failed",
            "raw_message": str(exc),
            "retryable": True,
        }
        node_events = partial_result.get("node_events") or telemetry_sink.get("node_events") or []
        tool_events = partial_result.get("tool_events") or telemetry_sink.get("tool_events") or []
        route = partial_result.get("route")
        route_reason = partial_result.get("route_reason")
        for event in reversed(node_events):
            if not isinstance(event, dict):
                continue
            if route is None and event.get("route"):
                route = event.get("route")
            if route_reason is None and event.get("route_reason"):
                route_reason = event.get("route_reason")
            if route is not None and route_reason is not None:
                break
        errors = [*partial_result.get("errors", []), error_payload]
        metadata = {
            **agent_run_context,
            "agent_route": route,
            "agent_route_reason": route_reason,
            "agent_node_events": node_events,
            "agent_tool_events": tool_events,
            "agent_error": error_payload,
        }
        turn = await create_chat_turn(
            thread_id=thread_id,
            question=req.question,
            answer=fallback_answer,
            status="failed",
            reasoning=f"Exception during Router RAG execution: {exc}",
            reasoning_available=True,
            reasoning_format="markdown",
            web_sources=[],
            document_sources=[],
            used_chat_ids=[],
            error=error_payload,
            metadata=metadata,
        )
        return {
            "answer": fallback_answer,
            "rewritten_query": question,
            "chat_turn_id": turn.id,
            "user_message_id": f"{turn.id}:user",
            "assistant_message_id": f"{turn.id}:assistant",
            "used_chat_ids": [],
            "document_sources": [],
            "web_sources": [],
            "clarification_options": None,
            "reasoning": f"Exception during Router RAG execution: {exc}",
            "reasoning_available": True,
            "reasoning_format": "markdown",
            "context": "Compiled Router RAG Agent execution failed gracefully.",
            "route": route,
            "route_reason": route_reason,
            "node_events": node_events,
            "tool_events": tool_events,
            "errors": errors,
            "duration_ms": duration_ms,
            "agent_error": error_payload,
            **agent_run_context,
        }
