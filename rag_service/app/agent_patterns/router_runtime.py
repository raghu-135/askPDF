from __future__ import annotations

import time
import logging
from typing import Any, Dict

from langgraph.types import Command

from app.agent_patterns.graph import TemplateCompiler
from app.db import (
    create_chat_turn,
    increment_qa_stats,
    update_message_context_compact,
)
from app.rag.indexer import index_chat_memory_for_thread
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET
from app.agent_patterns.trace import compact_preview


logger = logging.getLogger(__name__)


async def _invoke_graph_with_partial_state(app: Any, graph_input: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    latest_state = dict(graph_input) if isinstance(graph_input, dict) else {}
    async for chunk in app.astream(graph_input, config=config, stream_mode="values"):
        if isinstance(chunk, dict):
            latest_state = chunk
    return latest_state


def _runtime_config(
    *,
    app_thread_id: str,
    checkpoint_thread_id: str,
    embed_model: Any = None,
    context_window: Any = None,
    use_web_search: Any = None,
    use_reranker: Any = None,
    telemetry_sink: Dict[str, Any],
    trace_recorder: Any = None,
) -> Dict[str, Any]:
    configurable = {
        "thread_id": checkpoint_thread_id,
        "checkpoint_thread_id": checkpoint_thread_id,
        "app_thread_id": app_thread_id,
        "telemetry_sink": telemetry_sink,
        "trace_recorder": trace_recorder,
    }
    if embed_model is not None:
        configurable["embedding_model"] = embed_model
    if context_window is not None:
        configurable["context_window"] = context_window
    if use_web_search is not None:
        configurable["use_web_search"] = use_web_search
    if use_reranker is not None:
        configurable["use_reranker"] = use_reranker
    return {"configurable": configurable}


def _first_interrupt(result: Dict[str, Any]) -> Any:
    interrupts = result.get("__interrupt__")
    if isinstance(interrupts, (list, tuple)) and interrupts:
        return interrupts[0]
    return interrupts


def _pending_interrupt_from_result(
    result: Dict[str, Any],
    *,
    checkpoint_thread_id: str,
) -> Dict[str, Any] | None:
    interrupt_obj = _first_interrupt(result)
    if not interrupt_obj:
        return None
    value = getattr(interrupt_obj, "value", None)
    payload = dict(value) if isinstance(value, dict) else {"prompt": str(value or "Human review requested.")}
    interrupt_id = getattr(interrupt_obj, "id", None)
    if interrupt_id:
        payload["interrupt_id"] = str(interrupt_id)
    payload["checkpoint_resume"] = True
    payload["checkpoint_thread_id"] = checkpoint_thread_id
    return payload


def _without_runtime_keys(result: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(result)
    cleaned.pop("__interrupt__", None)
    return cleaned


def _as_resume_action(interrupt: Dict[str, Any]) -> Any:
    decision = interrupt.get("decision") if isinstance(interrupt.get("decision"), dict) else {}
    return decision.get("action") or decision.get("requested_action") or interrupt.get("default_action")


def _interrupted_node_event(partial: Dict[str, Any], pending_interrupt: Dict[str, Any]) -> Dict[str, Any]:
    node_id = str(pending_interrupt.get("node_id") or pending_interrupt.get("gate_id") or "hitl_gate")
    return {
        "node": node_id,
        "status": "interrupted",
        "route": partial.get("route"),
        "route_reason": partial.get("route_reason"),
        "input_preview": {
            "question": compact_preview(partial.get("question")),
            "title": compact_preview(pending_interrupt.get("title")),
            "prompt": compact_preview(pending_interrupt.get("prompt") or pending_interrupt.get("body")),
            "input_summary": pending_interrupt.get("input_summary"),
        },
        "output_preview": {
            "interrupt_id": pending_interrupt.get("interrupt_id"),
            "gate_id": pending_interrupt.get("gate_id"),
            "type": pending_interrupt.get("type"),
            "mode": pending_interrupt.get("mode"),
            "phase": pending_interrupt.get("phase"),
            "target_node_id": pending_interrupt.get("target_node_id"),
            "allowed_actions": pending_interrupt.get("allowed_actions"),
            "default_action": pending_interrupt.get("default_action"),
            "options": pending_interrupt.get("options"),
            "proposed_tool": pending_interrupt.get("proposed_tool"),
            "proposed_final_answer": pending_interrupt.get("proposed_final_answer"),
        },
    }


def _node_events_with_interrupted_gate(
    *,
    partial: Dict[str, Any],
    telemetry_sink: Dict[str, Any],
    pending_interrupt: Dict[str, Any],
    trace_recorder: Any = None,
) -> list[Dict[str, Any]]:
    node_events = list(partial.get("node_events") or telemetry_sink.get("node_events") or [])
    interrupted = _interrupted_node_event(partial, pending_interrupt)
    node_id = interrupted["node"]
    has_interrupted = any(
        isinstance(event, dict)
        and event.get("node") == node_id
        and event.get("status") == "interrupted"
        for event in node_events
    )
    if not has_interrupted:
        node_events.append(interrupted)
        if trace_recorder is not None and hasattr(trace_recorder, "record_node_event"):
            trace_recorder.record_node_event(interrupted)
    return node_events


async def _persist_success_turn(
    *,
    thread_id: str,
    question: str,
    result: Dict[str, Any],
    agent_run_context: Dict[str, Any],
    duration_ms: float,
    success_context: str,
) -> Dict[str, Any]:
    answer = result.get("final_answer") or "I was unable to compose an answer. Please try rephrasing your question."
    clarification_options = result.get("clarification_options")
    status = "clarification" if clarification_options else "completed"
    embed_model = result.get("embedding_model")
    llm_model = result.get("llm_model")
    context_window = result.get("context_window") or DEFAULT_TOKEN_BUDGET
    metadata = {
        "agent_pattern_id": agent_run_context.get("agent_pattern_id"),
        "agent_pattern_version": agent_run_context.get("agent_pattern_version"),
        "agent_pattern_template_version_id": agent_run_context.get("agent_pattern_template_version_id"),
        "agent_route": result.get("route"),
        "agent_route_reason": result.get("route_reason"),
    }
    turn = await create_chat_turn(
        thread_id=thread_id,
        question=question,
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
        agent_run_id=agent_run_context.get("agent_run_id"),
        agent_run_turn_kind="assistant_final",
        agent_run_sequence=0,
        agent_trace_refs_json=None,
    )

    if not clarification_options and embed_model and llm_model:
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
        await increment_qa_stats(thread_id, len(question or "") + len(answer or ""))
    except Exception:
        pass

    return {
        "answer": answer,
        "rewritten_query": question,
        "chat_turn_id": turn.id,
        "user_message_id": f"{turn.id}:user",
        "assistant_message_id": f"{turn.id}:assistant",
        "used_chat_ids": result.get("used_chat_ids") or [],
        "document_sources": result.get("document_sources") or [],
        "web_sources": result.get("web_sources") or [],
        "clarification_options": clarification_options,
        "reasoning": result.get("reasoning") or "",
        "reasoning_available": bool(result.get("reasoning_available")),
        "reasoning_format": result.get("reasoning_format") or "none",
        "context": success_context,
        "route": result.get("route"),
        "route_reason": result.get("route_reason"),
        "node_events": result.get("node_events") or [],
        "tool_events": result.get("tool_events") or [],
        "duration_ms": duration_ms,
        "status": status,
        "agent_run_turn_kind": "assistant_final",
        "agent_run_sequence": 0,
        "agent_trace_refs": None,
        **agent_run_context,
    }


async def handle_router_rag_chat(
    thread_id: str,
    req: Any,
    embed_model: str,
    *,
    resolved_spec: Dict[str, Any],
    agent_run_context: Dict[str, Any],
    trace_recorder: Any,
    checkpointer: Any = None,
) -> Dict[str, Any]:
    """Execute the compiled Router RAG v2 graph and persist a chat turn."""
    return await _handle_compiled_rag_chat(
        thread_id,
        req,
        embed_model,
        resolved_spec=resolved_spec,
        agent_run_context=agent_run_context,
        trace_recorder=trace_recorder,
        checkpointer=checkpointer,
        runtime_label="Router RAG",
        failure_code="router_rag_execution_failed",
        failure_reason_prefix="Exception during Router RAG execution",
        success_context="Context retrieved by compiled Router RAG Agent pattern.",
        failure_context="Compiled Router RAG Agent execution failed gracefully.",
    )


async def handle_plan_execute_rag_chat(
    thread_id: str,
    req: Any,
    embed_model: str,
    *,
    resolved_spec: Dict[str, Any],
    agent_run_context: Dict[str, Any],
    trace_recorder: Any,
    checkpointer: Any = None,
) -> Dict[str, Any]:
    """Execute the compiled Plan-and-Execute RAG graph and persist a chat turn."""
    return await _handle_compiled_rag_chat(
        thread_id,
        req,
        embed_model,
        resolved_spec=resolved_spec,
        agent_run_context=agent_run_context,
        trace_recorder=trace_recorder,
        checkpointer=checkpointer,
        runtime_label="Plan-and-Execute RAG",
        failure_code="plan_execute_rag_execution_failed",
        failure_reason_prefix="Exception during Plan-and-Execute RAG execution",
        success_context="Context retrieved by compiled Plan-and-Execute RAG Agent pattern.",
        failure_context="Compiled Plan-and-Execute RAG Agent execution failed gracefully.",
    )


async def handle_evaluator_replanner_rag_chat(
    thread_id: str,
    req: Any,
    embed_model: str,
    *,
    resolved_spec: Dict[str, Any],
    agent_run_context: Dict[str, Any],
    trace_recorder: Any,
    checkpointer: Any = None,
) -> Dict[str, Any]:
    """Execute the compiled Evaluator/Replanner RAG graph and persist a chat turn."""
    return await _handle_compiled_rag_chat(
        thread_id,
        req,
        embed_model,
        resolved_spec=resolved_spec,
        agent_run_context=agent_run_context,
        trace_recorder=trace_recorder,
        checkpointer=checkpointer,
        runtime_label="Evaluator/Replanner RAG",
        failure_code="evaluator_replanner_rag_execution_failed",
        failure_reason_prefix="Exception during Evaluator/Replanner RAG execution",
        success_context="Context retrieved by compiled Evaluator/Replanner RAG Agent pattern.",
        failure_context="Compiled Evaluator/Replanner RAG Agent execution failed gracefully.",
    )


async def _handle_compiled_rag_chat(
    thread_id: str,
    req: Any,
    embed_model: str,
    *,
    resolved_spec: Dict[str, Any],
    agent_run_context: Dict[str, Any],
    trace_recorder: Any,
    checkpointer: Any,
    runtime_label: str,
    failure_code: str,
    failure_reason_prefix: str,
    success_context: str,
    failure_context: str,
) -> Dict[str, Any]:
    """Execute a compiled RAG graph and persist a chat turn."""

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
    hitl_policy = pattern_config.get("hitl_policy") if isinstance(pattern_config.get("hitl_policy"), dict) else {}
    loop_policy = pattern_config.get("loop_policy") if isinstance(pattern_config.get("loop_policy"), dict) else {}
    try:
        replans = max(1, int(pattern_config.get("replans", 1)))
    except (TypeError, ValueError):
        replans = 1
    checkpoint_thread_id = str(agent_run_context.get("checkpoint_thread_id") or agent_run_id or thread_id)

    started = time.perf_counter()
    app = TemplateCompiler().compile(
        resolved_spec,
        checkpointer=checkpointer,
    )
    telemetry_sink: Dict[str, Any] = {"node_events": [], "tool_events": []}
    config = _runtime_config(
        app_thread_id=thread_id,
        checkpoint_thread_id=checkpoint_thread_id,
        embed_model=embed_model,
        context_window=context_window,
        use_web_search=use_web_search,
        use_reranker=use_reranker,
        telemetry_sink=telemetry_sink,
        trace_recorder=trace_recorder,
    )
    state = {
        "agent_run_id": agent_run_id,
        "pattern_type": resolved_spec.get("pattern_type"),
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
        "hitl_policy": hitl_policy,
        "loop_policy": loop_policy,
        "node_visit_counts": {},
        "node_visit_sequence": [],
        "evidence_packets": [],
        "hitl_interrupt_counts": {},
        "replans": replans,
        "replan_count": 0,
        "replan_history": [],
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
            "%s run started | run_id=%s thread_id=%s pattern=%s version=%s question_chars=%s",
            runtime_label,
            agent_run_id,
            thread_id,
            agent_run_context.get("agent_pattern_id"),
            agent_run_context.get("agent_pattern_version"),
            len(question or ""),
        )
        result = await _invoke_graph_with_partial_state(app, state, config)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        pending_interrupt = _pending_interrupt_from_result(
            result,
            checkpoint_thread_id=checkpoint_thread_id,
        )
        if pending_interrupt:
            partial = _without_runtime_keys(result)
            node_events = _node_events_with_interrupted_gate(
                partial=partial,
                telemetry_sink=telemetry_sink,
                pending_interrupt=pending_interrupt,
                trace_recorder=trace_recorder,
            )
            logger.info(
                "%s run awaiting human | run_id=%s thread_id=%s route=%s elapsed_ms=%.1f",
                runtime_label,
                agent_run_id,
                thread_id,
                partial.get("route"),
                duration_ms,
            )
            return {
                "answer": partial.get("final_answer"),
                "rewritten_query": question,
                "used_chat_ids": partial.get("used_chat_ids") or [],
                "document_sources": partial.get("document_sources") or [],
                "web_sources": partial.get("web_sources") or [],
                "clarification_options": partial.get("clarification_options"),
                "reasoning": partial.get("reasoning") or "",
                "reasoning_available": bool(partial.get("reasoning_available")),
                "reasoning_format": partial.get("reasoning_format") or "none",
                "context": "Compiled agent execution paused for human review.",
                "route": partial.get("route"),
                "route_reason": partial.get("route_reason"),
                "node_events": node_events,
                "tool_events": partial.get("tool_events") or [],
                "duration_ms": duration_ms,
                "status": "awaiting_human",
                "pending_interrupt": pending_interrupt,
                "agent_trace_refs": {"interrupt_id": pending_interrupt.get("interrupt_id")},
                **agent_run_context,
            }

        result = _without_runtime_keys(result)
        payload = await _persist_success_turn(
            thread_id=thread_id,
            question=question,
            result=result,
            agent_run_context=agent_run_context,
            duration_ms=duration_ms,
            success_context=success_context,
        )

        logger.info(
            "%s run completed | run_id=%s thread_id=%s route=%s status=%s elapsed_ms=%.1f document_sources=%s web_sources=%s used_chat_ids=%s node_events=%s tool_events=%s",
            runtime_label,
            agent_run_id,
            thread_id,
            result.get("route"),
            payload.get("status"),
            duration_ms,
            len(result.get("document_sources") or []),
            len(result.get("web_sources") or []),
            len(result.get("used_chat_ids") or []),
            len(result.get("node_events") or []),
            len(result.get("tool_events") or []),
        )

        return payload
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        partial_result = result if isinstance(locals().get("result"), dict) else state
        logger.exception(
            "%s run failed | run_id=%s thread_id=%s elapsed_ms=%.1f",
            runtime_label,
            agent_run_id,
            thread_id,
            duration_ms,
        )
        fallback_answer = (
            "I'm sorry, I encountered a technical error while processing your request. "
            "Please try again in a moment or try rephrasing your question."
        )
        error_payload = {
            "code": failure_code,
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
        failure_result = {
            **partial_result,
            "node_events": node_events,
            "tool_events": tool_events,
            "route": route,
            "route_reason": route_reason,
            "errors": errors,
            "agent_error": error_payload,
        }
        metadata = {
            "agent_pattern_id": agent_run_context.get("agent_pattern_id"),
            "agent_pattern_version": agent_run_context.get("agent_pattern_version"),
            "agent_pattern_template_version_id": agent_run_context.get("agent_pattern_template_version_id"),
            "agent_route": route,
            "agent_route_reason": route_reason,
            "agent_error": error_payload,
        }
        turn = await create_chat_turn(
            thread_id=thread_id,
            question=req.question,
            answer=fallback_answer,
            status="failed",
            reasoning=f"{failure_reason_prefix}: {exc}",
            reasoning_available=True,
            reasoning_format="markdown",
            web_sources=[],
            document_sources=[],
            used_chat_ids=[],
            error=error_payload,
            metadata=metadata,
            agent_run_id=agent_run_id,
            agent_run_turn_kind="assistant_final",
            agent_run_sequence=0,
            agent_trace_refs_json=None,
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
            "reasoning": f"{failure_reason_prefix}: {exc}",
            "reasoning_available": True,
            "reasoning_format": "markdown",
            "context": failure_context,
            "route": route,
            "route_reason": route_reason,
            "node_events": node_events,
            "tool_events": tool_events,
            "errors": errors,
            "duration_ms": duration_ms,
            "status": "failed",
            "agent_error": error_payload,
            "agent_run_turn_kind": "assistant_final",
            "agent_run_sequence": 0,
            "agent_trace_refs": None,
            **agent_run_context,
        }


async def resume_compiled_rag_chat(
    run: Any,
    *,
    interrupt: Dict[str, Any],
    checkpointer: Any,
    trace_recorder: Any = None,
) -> Dict[str, Any]:
    """Resume a checkpointed compiled RAG graph and persist the final chat turn."""

    resolved_spec = run.resolved_spec_json if isinstance(run.resolved_spec_json, dict) else {}
    checkpoint_thread_id = str(run.checkpoint_thread_id or run.id)
    telemetry_sink: Dict[str, Any] = {"node_events": [], "tool_events": []}
    app = TemplateCompiler().compile(
        resolved_spec,
        checkpointer=checkpointer,
    )
    config = _runtime_config(
        app_thread_id=run.thread_id,
        checkpoint_thread_id=checkpoint_thread_id,
        telemetry_sink=telemetry_sink,
        trace_recorder=None,
    )
    snapshot = await app.aget_state(config)
    snapshot_values = dict(getattr(snapshot, "values", None) or {})
    config = _runtime_config(
        app_thread_id=run.thread_id,
        checkpoint_thread_id=checkpoint_thread_id,
        embed_model=snapshot_values.get("embedding_model"),
        context_window=snapshot_values.get("context_window") or DEFAULT_TOKEN_BUDGET,
        use_web_search=snapshot_values.get("use_web_search"),
        use_reranker=snapshot_values.get("use_reranker"),
        telemetry_sink=telemetry_sink,
        trace_recorder=trace_recorder,
    )
    decision = interrupt.get("decision") if isinstance(interrupt.get("decision"), dict) else {}
    agent_run_context = {
        "agent_run_id": run.id,
        "agent_pattern_id": run.template_id,
        "agent_pattern_version": run.template_version,
        "agent_pattern_template_version_id": run.template_version_id,
        "checkpoint_thread_id": checkpoint_thread_id,
    }

    started = time.perf_counter()
    if trace_recorder is not None and hasattr(trace_recorder, "record_runtime_event"):
        trace_recorder.record_runtime_event(
            "graph.resumed",
            attributes={
                "askpdf.run.id": run.id,
                "askpdf.thread.id": run.thread_id,
                "askpdf.interrupt.id": interrupt.get("interrupt_id"),
                "askpdf.resume.action": _as_resume_action(interrupt),
                "askpdf.checkpoint.thread_id": checkpoint_thread_id,
            },
        )
    result = await _invoke_graph_with_partial_state(app, Command(resume=decision), config)
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    pending_interrupt = _pending_interrupt_from_result(
        result,
        checkpoint_thread_id=checkpoint_thread_id,
    )
    if pending_interrupt:
        partial = _without_runtime_keys(result)
        node_events = _node_events_with_interrupted_gate(
            partial=partial,
            telemetry_sink=telemetry_sink,
            pending_interrupt=pending_interrupt,
            trace_recorder=trace_recorder,
        )
        return {
            **partial,
            "node_events": node_events,
            "status": "awaiting_human",
            "pending_interrupt": pending_interrupt,
            "duration_ms": duration_ms,
            **agent_run_context,
        }

    result = _without_runtime_keys(result)
    question = str(result.get("question") or snapshot_values.get("question") or "")
    payload = await _persist_success_turn(
        thread_id=run.thread_id,
        question=question,
        result=result,
        agent_run_context=agent_run_context,
        duration_ms=duration_ms,
        success_context="Context retrieved by resumed compiled Agent pattern.",
    )
    logger.info(
        "Checkpointed agent run resumed | run_id=%s thread_id=%s route=%s status=%s elapsed_ms=%.1f",
        run.id,
        run.thread_id,
        payload.get("route"),
        payload.get("status"),
        duration_ms,
    )
    return payload
