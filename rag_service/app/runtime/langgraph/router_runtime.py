from __future__ import annotations

import time
import logging
from typing import Any, Dict

from langgraph.types import Command

from app.runtime.langgraph.compiler import WorkflowCompiler
from app.agent_workflows.chat_cancellation import (
    ChatRunCancellationRequested,
    raise_if_chat_run_cancelled,
)
from app.agent_workflows.enums import NodeEventStatus, WorkflowNodeType
from app.agent_workflows.planning import worker_nodes_from_spec
from app.agent_workflows.parallel_runtime import cancelled_parallel_dispatch, normalized_parallel_policy
from app.agent_workflows.parallel_contracts import ParallelEventName
from app.agent_workflows.corrective_contracts import CORRECTIVE_WORKFLOW_ID, normalized_corrective_policy
from app.agent_workflows.state import merge_parallel_deltas, WorkflowBudgetExceeded
from app.agent_workflows.workflow_runtime import runtime_execution_options, workflow_runtime_features
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET
from app.agent_workflows.trace import compact_preview


logger = logging.getLogger(__name__)


class _Status:
    RUNNING = type("Value", (), {"value": "running"})
    AWAITING_HUMAN = type("Value", (), {"value": "awaiting_human"})
    COMPLETED = type("Value", (), {"value": "completed"})
    FAILED = type("Value", (), {"value": "failed"})
    CANCELLED = type("Value", (), {"value": "cancelled"})
    MARKDOWN = type("Value", (), {"value": "markdown"})
    NONE = type("Value", (), {"value": "none"})


AgentRunStatus = ReasoningFormat = _Status


def _corrective_metrics_state(result: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("workflow_id") != CORRECTIVE_WORKFLOW_ID:
        return {}
    keys = (
        "workflow_id", "corrective_wave", "corrective_history", "corrective_budget_exhausted_reason",
        "corrective_termination_reason",
        "worker_result_packets", "retrieval_quality_report", "grounding_report", "corrective_policy_filtered_proposals",
    )
    return {key: result.get(key) for key in keys}


async def _invoke_graph_with_partial_state(app: Any, graph_input: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    latest_state = dict(graph_input) if isinstance(graph_input, dict) else {}
    try:
        async for chunk in app.astream(graph_input, config=config, stream_mode="values"):
            if isinstance(chunk, dict):
                latest_state = chunk
    except ChatRunCancellationRequested as exc:
        exc.state = {**latest_state, **dict(exc.state or {})}
        raise
    except Exception as exc:
        exc.agent_workflow_state = latest_state
        raise
    return latest_state


def _runtime_config(
    *,
    app_thread_id: str,
    checkpoint_thread_id: str,
    embedding_model: Any = None,
    context_window: Any = None,
    use_web_search: Any = None,
    use_reranker: Any = None,
    telemetry_sink: Dict[str, Any],
    trace_recorder: Any = None,
    execution_event_sink: Any = None,
    cancellation_checker: Any = None,
    pause_checker: Any = None,
    max_concurrency: int | None = None,
    deep_research_services_factory: Any,
) -> Dict[str, Any]:
    configurable = {
        "thread_id": checkpoint_thread_id,
        "checkpoint_thread_id": checkpoint_thread_id,
        "app_thread_id": app_thread_id,
        "telemetry_sink": telemetry_sink,
        "trace_recorder": trace_recorder,
        "deep_research_services_factory": deep_research_services_factory,
    }
    if execution_event_sink is not None:
        configurable["execution_event_sink"] = execution_event_sink
    if cancellation_checker is not None:
        configurable["cancellation_checker"] = cancellation_checker
    if pause_checker is not None:
        configurable["pause_checker"] = pause_checker
    if embedding_model is not None:
        configurable["embedding_model"] = embedding_model
    if context_window is not None:
        configurable["context_window"] = context_window
    if use_web_search is not None:
        configurable["use_web_search"] = use_web_search
    if use_reranker is not None:
        configurable["use_reranker"] = use_reranker
    result: Dict[str, Any] = {"configurable": configurable}
    if isinstance(max_concurrency, int) and max_concurrency > 0:
        result["max_concurrency"] = max_concurrency
    return result


def _deep_research_services_factory() -> Any:
    from app.agent_workflows.deep_research_execution import (
        runtime_execution_services_factory,
    )
    return runtime_execution_services_factory


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


def _runtime_result(
    result: Dict[str, Any],
    *,
    status: str,
    duration_ms: float,
    agent_run_context: Dict[str, Any],
    answer: str | None = None,
) -> Dict[str, Any]:
    """Return JSON-compatible execution output without product persistence."""

    return {
        **result,
        "answer": answer if answer is not None else result.get("final_answer") or result.get("answer") or "",
        "status": status,
        "duration_ms": duration_ms,
        **agent_run_context,
    }


def _as_resume_action(interrupt: Dict[str, Any]) -> Any:
    decision = interrupt.get("decision") if isinstance(interrupt.get("decision"), dict) else {}
    return decision.get("action") or decision.get("requested_action") or interrupt.get("default_action")


def _interrupted_node_event(partial: Dict[str, Any], pending_interrupt: Dict[str, Any]) -> Dict[str, Any]:
    node_id = str(pending_interrupt.get("node_id") or pending_interrupt.get("gate_id") or WorkflowNodeType.HITL_GATE.value)
    return {
        "node": node_id,
        "status": NodeEventStatus.INTERRUPTED.value,
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
        and event.get("status") == NodeEventStatus.INTERRUPTED.value
        for event in node_events
    )
    if not has_interrupted:
        node_events.append(interrupted)
        if trace_recorder is not None and hasattr(trace_recorder, "record_node_event"):
            trace_recorder.record_node_event(interrupted)
    return node_events


async def execute_compiled_rag_chat(
    thread_id: str,
    req: Any,
    embedding_model: str,
    *,
    resolved_spec: Dict[str, Any],
    agent_run_context: Dict[str, Any],
    trace_recorder: Any,
    checkpointer: Any = None,
    execution_event_sink: Any = None,
    cancellation_checker: Any = None,
    pause_checker: Any = None,
) -> Dict[str, Any]:
    """Execute a compiled RAG workflow using runtime metadata from the stored spec."""
    runtime_options = runtime_execution_options(resolved_spec)
    return await _handle_compiled_rag_chat(
        thread_id,
        req,
        embedding_model,
        resolved_spec=resolved_spec,
        agent_run_context=agent_run_context,
        trace_recorder=trace_recorder,
        checkpointer=checkpointer,
        execution_event_sink=execution_event_sink,
        cancellation_checker=cancellation_checker,
        pause_checker=pause_checker,
        runtime_label=runtime_options["label"],
        failure_code=runtime_options["failure_code"],
        failure_reason_prefix=runtime_options["failure_reason_prefix"],
        success_context=runtime_options["success_context"],
        failure_context=runtime_options["failure_context"],
    )


async def handle_router_rag_chat(
    thread_id: str,
    req: Any,
    embedding_model: str,
    *,
    resolved_spec: Dict[str, Any],
    agent_run_context: Dict[str, Any],
    trace_recorder: Any,
    checkpointer: Any = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Execute the Router workflow runtime."""

    return await execute_compiled_rag_chat(
        thread_id,
        req,
        embedding_model,
        resolved_spec=resolved_spec,
        agent_run_context=agent_run_context,
        trace_recorder=trace_recorder,
        checkpointer=checkpointer,
        execution_event_sink=_kwargs.get("execution_event_sink"),
        cancellation_checker=_kwargs.get("cancellation_checker"),
    )


async def handle_plan_execute_rag_chat(
    thread_id: str,
    req: Any,
    embedding_model: str,
    *,
    resolved_spec: Dict[str, Any],
    agent_run_context: Dict[str, Any],
    trace_recorder: Any,
    checkpointer: Any = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Execute the plan/execute RAG workflow runtime."""

    return await execute_compiled_rag_chat(
        thread_id,
        req,
        embedding_model,
        resolved_spec=resolved_spec,
        agent_run_context=agent_run_context,
        trace_recorder=trace_recorder,
        checkpointer=checkpointer,
        execution_event_sink=_kwargs.get("execution_event_sink"),
        cancellation_checker=_kwargs.get("cancellation_checker"),
    )


async def handle_evaluator_replanner_rag_chat(
    thread_id: str,
    req: Any,
    embedding_model: str,
    *,
    resolved_spec: Dict[str, Any],
    agent_run_context: Dict[str, Any],
    trace_recorder: Any,
    checkpointer: Any = None,
    **_kwargs: Any,
) -> Dict[str, Any]:
    """Execute the evaluator/replanner RAG workflow runtime."""

    return await execute_compiled_rag_chat(
        thread_id,
        req,
        embedding_model,
        resolved_spec=resolved_spec,
        agent_run_context=agent_run_context,
        trace_recorder=trace_recorder,
        checkpointer=checkpointer,
        execution_event_sink=_kwargs.get("execution_event_sink"),
        cancellation_checker=_kwargs.get("cancellation_checker"),
    )


async def _handle_compiled_rag_chat(
    thread_id: str,
    req: Any,
    embedding_model: str,
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
    execution_event_sink: Any = None,
    cancellation_checker: Any = None,
    pause_checker: Any = None,
) -> Dict[str, Any]:
    """Execute a compiled RAG graph and return runtime-owned output."""

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
    workflow_config = resolved_spec.get("config") if isinstance(resolved_spec.get("config"), dict) else {}
    allowed_tool_ids = workflow_config.get("allowed_tool_ids")
    allowed_tool_ids = allowed_tool_ids if isinstance(allowed_tool_ids, list) else []
    hitl_policy = workflow_config.get("hitl_policy") if isinstance(workflow_config.get("hitl_policy"), dict) else {}
    loop_policy = workflow_config.get("loop_policy") if isinstance(workflow_config.get("loop_policy"), dict) else {}
    context_policy = workflow_config.get("context_policy") if isinstance(workflow_config.get("context_policy"), dict) else {}
    prefetch_policy = workflow_config.get("prefetch_policy") if isinstance(workflow_config.get("prefetch_policy"), dict) else {}
    runtime_features = workflow_runtime_features(resolved_spec)
    parallel_enabled = bool(runtime_features.get("supports_parallel_dispatch"))
    parallel_policy = normalized_parallel_policy(workflow_config.get("parallel_policy"))
    corrective_policy = normalized_corrective_policy(workflow_config.get("corrective_policy"))
    graph_nodes = ((workflow_config.get("graph") or {}).get("nodes") or []) if isinstance(workflow_config.get("graph"), dict) else []
    parallel_aggregator_id = next(
        (
            str(node.get("id"))
            for node in graph_nodes
            if isinstance(node, dict) and node.get("type") == WorkflowNodeType.AGGREGATOR.value
        ),
        "",
    )
    try:
        replans = max(1, int(workflow_config.get("replans", 1)))
    except (TypeError, ValueError):
        replans = 1
    checkpoint_thread_id = str(agent_run_context.get("checkpoint_thread_id") or agent_run_id or thread_id)

    started = time.perf_counter()
    app = WorkflowCompiler().compile(
        resolved_spec,
        checkpointer=checkpointer,
    )
    telemetry_sink: Dict[str, Any] = {"node_events": [], "tool_events": []}
    config = _runtime_config(
        app_thread_id=thread_id,
        checkpoint_thread_id=checkpoint_thread_id,
        embedding_model=embedding_model,
        context_window=context_window,
        use_web_search=use_web_search,
        use_reranker=use_reranker,
        telemetry_sink=telemetry_sink,
        trace_recorder=trace_recorder,
        execution_event_sink=execution_event_sink,
        cancellation_checker=cancellation_checker,
        pause_checker=pause_checker,
        max_concurrency=parallel_policy["max_concurrency"] if parallel_enabled else None,
        deep_research_services_factory=_deep_research_services_factory(),
    )
    state = {
        "agent_run_id": agent_run_id,
        "workflow_id": resolved_spec.get("workflow_id"),
        "thread_id": thread_id,
        "question": question,
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "context_window": context_window,
        "use_web_search": use_web_search,
        "use_reranker": use_reranker,
        "bypass_clarification": bool(getattr(req, "bypass_clarification", False)),
        "system_role": system_role,
        "tool_instructions": tool_instructions,
        "custom_instructions": custom_instructions,
        "allowed_tool_ids": allowed_tool_ids,
        "available_worker_nodes": worker_nodes_from_spec(resolved_spec),
        "hitl_policy": hitl_policy,
        "loop_policy": loop_policy,
        "context_policy": context_policy,
        "prefetch_policy": prefetch_policy,
        "parallel_enabled": parallel_enabled,
        "parallel_policy": parallel_policy,
        "corrective_policy": corrective_policy,
        "corrective_wave": 0,
        "corrective_history": [],
        "corrective_wave_records": [],
        "corrective_policy_filtered_proposals": [],
        "corrective_budget_usage": {},
        "corrective_budget_exhausted_reason": "",
        "corrective_termination_reason": "",
        "retrieval_quality_report": {},
        "evidence_assessments": [],
        "source_assessments": [],
        "unresolved_gaps": [],
        "grounding_report": {},
        "verified_claims": [],
        "contradiction_report": [],
        "answer_revision_count": 0,
        "parallel_aggregator_id": parallel_aggregator_id,
        "dispatch_aggregator_id": parallel_aggregator_id,
        "worker_result_packets": [],
        "parallel_evidence_deltas": [],
        "parallel_document_source_deltas": [],
        "parallel_web_source_deltas": [],
        "parallel_chat_id_deltas": [],
        "parallel_memory_ref_deltas": [],
        "parallel_timeline_ref_deltas": [],
        "parallel_node_event_deltas": [],
        "parallel_tool_event_deltas": [],
        "parallel_error_deltas": [],
        "parallel_skipped_work_deltas": [],
        "parallel_visit_records": [],
        "parallel_attempt_records": [],
        "node_visit_counts": {},
        "node_visit_sequence": [],
        "evidence_packets": [],
        "hitl_interrupt_counts": {},
        "hitl_approval_grants": {},
        "replans": replans,
        "replan_count": 0,
        "replan_history": [],
        "client_timezone": getattr(req, "client_timezone", None),
        "client_locale": getattr(req, "client_locale", None),
        "client_now_iso": getattr(req, "client_now_iso", None),
        # Durable deep-research fields are opt-in. Ordinary workflows ignore
        # them, which keeps the established chat execution contract unchanged.
        "agent_task_id": getattr(req, "agent_task_id", None),
        "web_search_mode": str(getattr(req, "web_search_mode", "on" if use_web_search else "off")),
        "task_web_access": str(getattr(req, "task_web_access", "undecided")),
        "task_web_access_decision": {},
        "task_version": getattr(req, "agent_task_version", None),
        "task_enabled_profiles": list(getattr(req, "task_enabled_profiles", None) or []),
        "task_limits": dict(getattr(req, "task_limits", None) or {}),
        "task_plan_revision": int(getattr(req, "task_plan_revision", 0) or 0),
        "task_run_plan_count": int(getattr(req, "task_run_plan_count", 0) or 0),
        "task_plan": dict(getattr(req, "task_plan", None) or {}),
        "task_todos": list(getattr(req, "task_todos", None) or []),
        "task_work_items": [],
        "task_result_packets": [],
        "task_result_warnings": [],
        "task_result_gaps": [],
        "task_artifact_manifest": [],
        "task_evidence_manifest": [],
        "task_context_summary": {},
        "task_memory_snapshot": dict(getattr(req, "task_memory_snapshot", None) or {}),
        "task_budget_usage": dict(getattr(req, "task_budget_usage", None) or {}),
        "task_orchestration": dict(getattr(req, "task_orchestration", None) or {}),
        "runtime_execution_mode": bool(getattr(req, "runtime_execution_mode", False)),
        "runtime_artifact_manifest": list(getattr(req, "runtime_artifact_manifest", None) or []),
        "runtime_artifact_contents": dict(getattr(req, "runtime_artifact_contents", None) or {}),
        "runtime_artifacts": [],
        "task_incomplete_reasons": [],
        "task_pause_requested": False,
        "task_cancel_requested": False,
        "document_sources": [],
        "web_sources": [],
        "used_chat_ids": [],
        "used_memory_ids": [],
        "node_events": [],
        "tool_events": [],
        "errors": [],
    }

    try:
        logger.info(
            "%s run started | run_id=%s thread_id=%s workflow=%s question_chars=%s",
            runtime_label,
            agent_run_id,
            thread_id,
            agent_run_context.get("agent_workflow_id"),
            len(question or ""),
        )
        result = await _invoke_graph_with_partial_state(app, state, config)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        await raise_if_chat_run_cancelled(cancellation_checker, result)
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
                "reasoning_format": partial.get("reasoning_format") or ReasoningFormat.NONE.value,
                "context": "Compiled agent execution paused for human review.",
                "route": partial.get("route"),
                "route_reason": partial.get("route_reason"),
                "node_events": node_events,
                "tool_events": partial.get("tool_events") or [],
                "duration_ms": duration_ms,
                "status": AgentRunStatus.AWAITING_HUMAN.value,
                "pending_interrupt": pending_interrupt,
                "agent_trace_refs": {"interrupt_id": pending_interrupt.get("interrupt_id")},
                "parallel_summary": partial.get("parallel_summary"),
                "_parallel_attempt_records": partial.get("parallel_attempt_records") or [],
                "_corrective_wave_records": partial.get("corrective_wave_records") or [],
                "_corrective_metrics_state": _corrective_metrics_state(partial),
                **agent_run_context,
            }

        result = _without_runtime_keys(result)
        if result.get("clarification_options"):
            payload = _runtime_result(
                result,
                status="clarification_required",
                duration_ms=duration_ms,
                agent_run_context=agent_run_context,
            )
        else:
            payload = _runtime_result(
                result,
                status=AgentRunStatus.COMPLETED.value,
                duration_ms=duration_ms,
                agent_run_context=agent_run_context,
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
    except ChatRunCancellationRequested as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        partial_result = _without_runtime_keys(exc.state or state)
        parallel_events = (
            execution_event_sink.parallel_events()
            if execution_event_sink is not None and hasattr(execution_event_sink, "parallel_events")
            else []
        )
        if partial_result.get("parallel_enabled") and (partial_result.get("work_items") or parallel_events):
            cancellation_update = cancelled_parallel_dispatch(partial_result, parallel_events)
            partial_result.update(cancellation_update)
            if execution_event_sink is not None:
                for packet in cancellation_update.get("worker_result_packets", []):
                    await execution_event_sink.emit(ParallelEventName.WORKER_CANCELLED, {
                        "agent_run_id": agent_run_id,
                        "dispatch_id": packet.get("dispatch_id"),
                        "work_id": packet.get("work_id"),
                        "ordinal": packet.get("ordinal"),
                        "worker_node_id": packet.get("worker_node_id"),
                        "worker_type": packet.get("worker_type"),
                        "attempt": packet.get("attempt"),
                        "status": "cancelled",
                    })
                await execution_event_sink.emit(ParallelEventName.DISPATCH_CANCELLED, cancellation_update.get("parallel_summary") or {})
        partial_result["node_events"] = partial_result.get("node_events") or telemetry_sink.get("node_events") or []
        partial_result["tool_events"] = partial_result.get("tool_events") or telemetry_sink.get("tool_events") or []
        logger.info(
            "%s run canceled | run_id=%s thread_id=%s elapsed_ms=%.1f node_events=%s tool_events=%s",
            runtime_label,
            agent_run_id,
            thread_id,
            duration_ms,
            len(partial_result["node_events"]),
            len(partial_result["tool_events"]),
        )
        return _runtime_result(
            partial_result,
            status=AgentRunStatus.CANCELLED.value,
            duration_ms=duration_ms,
            agent_run_context=agent_run_context,
            answer="",
        )
    except WorkflowBudgetExceeded as exc:
        partial = _without_runtime_keys(getattr(exc, "agent_workflow_state", None) or snapshot_values)
        partial["workflow_budget"] = exc.as_dict()
        partial["agent_error"] = {"code": "workflow_budget_exhausted", "retryable": False, "partial": True, "workflow_budget": exc.as_dict()}
        return {**partial, "status": AgentRunStatus.FAILED.value, "duration_ms": round((time.perf_counter() - started) * 1000, 2), **agent_run_context}
    except Exception as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        exception_state = getattr(exc, "agent_workflow_state", None)
        partial_result = (
            exception_state
            if isinstance(exception_state, dict)
            else result
            if isinstance(locals().get("result"), dict)
            else state
        )
        budget_exhausted = isinstance(exc, WorkflowBudgetExceeded)
        if budget_exhausted:
            logger.warning(
                "%s run bounded by workflow visit budget | run_id=%s thread_id=%s limit=%s node=%s elapsed_ms=%.1f",
                runtime_label, agent_run_id, thread_id, exc.limit, exc.node_id, duration_ms,
            )
        else:
            logger.exception(
                "%s run failed | run_id=%s thread_id=%s elapsed_ms=%.1f",
                runtime_label, agent_run_id, thread_id, duration_ms,
            )
        fallback_answer = (
            "I reached the workflow execution limit before completing the answer. "
            "The evidence collected so far is preserved; please retry with a narrower question."
            if budget_exhausted else
            "I'm sorry, I encountered a technical error while processing your request. "
            "Please try again in a moment or try rephrasing your question."
        )
        parallel_unauthorized = str(exc) == "parallel runtime is not authorized for this workflow"
        error_payload = {
            "code": "workflow_budget_exhausted" if budget_exhausted else ("agent_workflow_parallel_unauthorized" if parallel_unauthorized else failure_code),
            "raw_message": str(exc),
            "retryable": False if budget_exhausted else not parallel_unauthorized,
            "partial": budget_exhausted,
        }
        if budget_exhausted:
            error_payload["workflow_budget"] = exc.as_dict()
        node_events = merge_parallel_deltas(
            partial_result.get("node_events") or telemetry_sink.get("node_events") or [],
            partial_result.get("parallel_node_event_deltas") or [],
        )
        tool_events = merge_parallel_deltas(
            partial_result.get("tool_events") or telemetry_sink.get("tool_events") or [],
            partial_result.get("parallel_tool_event_deltas") or [],
        )
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
        errors = [
            *(partial_result.get("errors") or partial_result.get("parallel_error_deltas") or []),
            error_payload,
        ]
        failure_result = {
            **partial_result,
            "node_events": node_events,
            "tool_events": tool_events,
            "route": route,
            "route_reason": route_reason,
            "errors": errors,
            "agent_error": error_payload,
            **({"workflow_budget": exc.as_dict()} if budget_exhausted else {}),
        }
        return {
            **failure_result,
            "answer": fallback_answer,
            "used_chat_ids": [],
            "document_sources": [],
            "web_sources": [],
            "clarification_options": None,
            "reasoning": f"{failure_reason_prefix}: {exc}",
            "reasoning_available": True,
            "reasoning_format": ReasoningFormat.MARKDOWN.value,
            "context": failure_context,
            "route": route,
            "route_reason": route_reason,
            "node_events": node_events,
            "tool_events": tool_events,
            "errors": errors,
            "duration_ms": duration_ms,
            "status": NodeEventStatus.FAILED.value,
            "agent_error": error_payload,
            "parallel_summary": partial_result.get("parallel_summary"),
            "_parallel_attempt_records": partial_result.get("parallel_attempt_records") or [],
            "_corrective_wave_records": partial_result.get("corrective_wave_records") or [],
            "_corrective_metrics_state": _corrective_metrics_state(partial_result),
            **agent_run_context,
        }


async def continue_compiled_rag_chat(
    run: Any,
    *,
    checkpointer: Any,
    trace_recorder: Any = None,
    execution_event_sink: Any = None,
    cancellation_checker: Any = None,
    pause_checker: Any = None,
) -> Dict[str, Any] | None:
    """Continue a nonterminal graph from its latest durable checkpoint."""

    resolved_spec = run.resolved_spec_json if isinstance(run.resolved_spec_json, dict) else {}
    checkpoint_thread_id = str(run.checkpoint_thread_id or run.id)
    telemetry_sink: Dict[str, Any] = {"node_events": [], "tool_events": []}
    app = WorkflowCompiler().compile(resolved_spec, checkpointer=checkpointer)
    initial_config = _runtime_config(
        app_thread_id=run.thread_id,
        checkpoint_thread_id=checkpoint_thread_id,
        telemetry_sink=telemetry_sink,
        trace_recorder=None,
        deep_research_services_factory=_deep_research_services_factory(),
    )
    snapshot = await app.aget_state(initial_config)
    snapshot_values = dict(getattr(snapshot, "values", None) or {})
    if not snapshot_values:
        return None
    config = _runtime_config(
        app_thread_id=run.thread_id,
        checkpoint_thread_id=checkpoint_thread_id,
        embedding_model=snapshot_values.get("embedding_model"),
        context_window=snapshot_values.get("context_window") or DEFAULT_TOKEN_BUDGET,
        use_web_search=snapshot_values.get("use_web_search"),
        use_reranker=snapshot_values.get("use_reranker"),
        telemetry_sink=telemetry_sink,
        trace_recorder=trace_recorder,
        execution_event_sink=execution_event_sink,
        cancellation_checker=cancellation_checker,
        pause_checker=pause_checker,
        deep_research_services_factory=_deep_research_services_factory(),
    )
    agent_run_context = {
        "agent_run_id": run.id,
        "agent_workflow_id": run.workflow_id,
        "checkpoint_thread_id": checkpoint_thread_id,
    }
    started = time.perf_counter()
    if trace_recorder is not None and hasattr(trace_recorder, "record_runtime_event"):
        trace_recorder.record_runtime_event(
            "graph.continued",
            attributes={
                "askpdf.run.id": run.id,
                "askpdf.thread.id": run.thread_id,
                "askpdf.checkpoint.thread_id": checkpoint_thread_id,
            },
        )
    try:
        if getattr(snapshot, "next", None):
            result = await _invoke_graph_with_partial_state(app, None, config)
        else:
            result = snapshot_values
        await raise_if_chat_run_cancelled(cancellation_checker, result)
    except ChatRunCancellationRequested as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        partial_result = _without_runtime_keys(exc.state or snapshot_values)
        return _runtime_result(
            partial_result,
            status=AgentRunStatus.CANCELLED.value,
            duration_ms=duration_ms,
            agent_run_context=agent_run_context,
            answer="",
        )
    except WorkflowBudgetExceeded as exc:
        partial = _without_runtime_keys(getattr(exc, "agent_workflow_state", None) or snapshot_values)
        partial["workflow_budget"] = exc.as_dict()
        partial["agent_error"] = {"code": "workflow_budget_exhausted", "retryable": False, "partial": True, "workflow_budget": exc.as_dict()}
        return {**partial, "status": AgentRunStatus.FAILED.value, "duration_ms": round((time.perf_counter() - started) * 1000, 2), **agent_run_context}
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    pending_interrupt = _pending_interrupt_from_result(result, checkpoint_thread_id=checkpoint_thread_id)
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
            "status": AgentRunStatus.AWAITING_HUMAN.value,
            "pending_interrupt": pending_interrupt,
            "duration_ms": duration_ms,
            **agent_run_context,
        }
    result = _without_runtime_keys(result)
    question = str(result.get("question") or snapshot_values.get("question") or "")
    if result.get("clarification_options"):
        payload = _runtime_result(
            result,
            status="clarification_required",
            duration_ms=duration_ms,
            agent_run_context=agent_run_context,
        )
    else:
        payload = _runtime_result(
            result,
            status=AgentRunStatus.COMPLETED.value,
            duration_ms=duration_ms,
            agent_run_context=agent_run_context,
        )
    logger.info(
        "Checkpointed agent run continued | run_id=%s thread_id=%s status=%s elapsed_ms=%.1f",
        run.id,
        run.thread_id,
        payload.get("status"),
        duration_ms,
    )
    return payload


async def resume_compiled_rag_chat(
    run: Any,
    *,
    interrupt: Dict[str, Any],
    checkpointer: Any,
    trace_recorder: Any = None,
    execution_event_sink: Any = None,
    cancellation_checker: Any = None,
    pause_checker: Any = None,
) -> Dict[str, Any]:
    """Resume a checkpointed compiled RAG graph and return runtime output."""

    resolved_spec = run.resolved_spec_json if isinstance(run.resolved_spec_json, dict) else {}
    checkpoint_thread_id = str(run.checkpoint_thread_id or run.id)
    telemetry_sink: Dict[str, Any] = {"node_events": [], "tool_events": []}
    app = WorkflowCompiler().compile(
        resolved_spec,
        checkpointer=checkpointer,
    )
    config = _runtime_config(
        app_thread_id=run.thread_id,
        checkpoint_thread_id=checkpoint_thread_id,
        telemetry_sink=telemetry_sink,
        trace_recorder=None,
        deep_research_services_factory=_deep_research_services_factory(),
    )
    snapshot = await app.aget_state(config)
    snapshot_values = dict(getattr(snapshot, "values", None) or {})
    config = _runtime_config(
        app_thread_id=run.thread_id,
        checkpoint_thread_id=checkpoint_thread_id,
        embedding_model=snapshot_values.get("embedding_model"),
        context_window=snapshot_values.get("context_window") or DEFAULT_TOKEN_BUDGET,
        use_web_search=snapshot_values.get("use_web_search"),
        use_reranker=snapshot_values.get("use_reranker"),
        telemetry_sink=telemetry_sink,
        trace_recorder=trace_recorder,
        execution_event_sink=execution_event_sink,
        cancellation_checker=cancellation_checker,
        pause_checker=pause_checker,
        deep_research_services_factory=_deep_research_services_factory(),
    )
    decision = interrupt.get("decision") if isinstance(interrupt.get("decision"), dict) else {}
    agent_run_context = {
        "agent_run_id": run.id,
        "agent_workflow_id": run.workflow_id,
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
    try:
        result = await _invoke_graph_with_partial_state(app, Command(resume=decision), config)
        await raise_if_chat_run_cancelled(cancellation_checker, result)
    except ChatRunCancellationRequested as exc:
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        partial_result = _without_runtime_keys(exc.state or snapshot_values)
        if partial_result.get("parallel_enabled") and partial_result.get("work_items"):
            partial_result.update(cancelled_parallel_dispatch(partial_result, []))
        return _runtime_result(
            partial_result,
            status=AgentRunStatus.CANCELLED.value,
            duration_ms=duration_ms,
            agent_run_context=agent_run_context,
            answer="",
        )
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
            "status": AgentRunStatus.AWAITING_HUMAN.value,
            "pending_interrupt": pending_interrupt,
            "duration_ms": duration_ms,
            **agent_run_context,
        }

    result = _without_runtime_keys(result)
    question = str(result.get("question") or snapshot_values.get("question") or "")
    if result.get("clarification_options"):
        payload = _runtime_result(
            result,
            status="clarification_required",
            duration_ms=duration_ms,
            agent_run_context=agent_run_context,
        )
    else:
        payload = _runtime_result(
            result,
            status=AgentRunStatus.COMPLETED.value,
            duration_ms=duration_ms,
            agent_run_context=agent_run_context,
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
