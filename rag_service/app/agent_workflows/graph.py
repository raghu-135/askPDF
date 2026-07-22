from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Any, Callable, Dict

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

from app.agent.tool_contract import normalize_tool_result
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET, get_llm
from app.agent.external_research_tools import search_web
from app.rag.agent_tools import search_conversation_history, search_documents, search_thread_timeline
from app.rag.chat_service import prefetch_context
from app.agent_workflows.prompting import (
    build_evaluator_prompt,
    build_planner_prompt,
    build_replanner_prompt,
    build_router_prompt,
)
from app.agent_workflows.answer_nodes import (
    direct_answer_node,
    final_context_from_state as _final_context_from_state,
    finalizer_node,
    synthesizer_node,
)
from app.agent_workflows.decision_nodes import (
    JsonDecisionNodeSpec,
    build_decision_node_event_data,
    invoke_json_decision_node,
)
from app.agent_workflows.evidence import (
    append_evidence_packet as _append_evidence_packet,
    combine_evidence as _combine_evidence,
    evidence_text_limit as _evidence_text_limit,
    format_prefetch_summary as _format_prefetch_summary,
    prefetch_refs as _prefetch_refs,
    state_evidence_refs as _state_evidence_refs,
)
from app.agent_workflows.enums import EvaluatorRoute, NodeEventStatus, RouterRoute, ROUTER_ROUTES, WorkflowNodeType
from app.agent_workflows.hitl_runtime import (
    WEB_APPROVAL_GATE_ID,
    hitl_gate_node,
    normalize_hitl_policy_for_thread_settings,
)
from app.agent_workflows.planning import (
    WORKER_NODE_ORDER,
    bounded_string_list as _bounded_string_list,
    current_replan_count as _current_replan_count,
    fallback_clarification_options as _fallback_clarification_options,
    infer_required_plan_steps,
    normalize_evaluator_report,
    normalize_execution_plan,
    normalize_replanner_execution_plan as _normalize_replanner_execution_plan,
    replan_budget as _replan_budget,
)
from app.agent_workflows.node_catalog import (
    get_node_type_metadata,
    node_type_capabilities,
)
from app.agent_workflows.runtime_invocation import (
    append_event as _append_event,
    append_failed_node_event as _append_failed_node_event,
    append_tool_event_for_node as _append_tool_event,
    invoke_llm_for_node as _invoke_llm_for_node,
    invoke_tool_for_node as _invoke_tool_for_node,
    llm_result_metadata as _llm_result_metadata,
    llm_retry_observer as _llm_retry_observer,
    log_node_end as _log_node_end,
    safe_json_object as _safe_json_object,
    should_skip_worker as _should_skip_worker,
    skipped_worker_update as _skipped_worker_update,
    tool_config as _tool_config,
    tool_config_for_node as _tool_config_for_node,
)
from app.agent_workflows.workers import run_tool_worker, tool_worker_spec
from app.agent_workflows.routes import (
    evaluator_route,
    hitl_gate_route,
    hitl_gate_route_for,
    planner_route,
    route_function_for_edge as _route_function_for_edge,
    router_route,
)
from app.agent_workflows.trace import (
    compact_preview,
    compact_refs,
    normalize_warnings,
    prompt_summary,
    refs_from_artifacts,
    selected_and_skipped_workers,
)
from app.agent_workflows.state import (
    RouterRagState,
    check_visit_budget as _check_visit_budget,
    node_visit_counts as _node_visit_counts,
    runtime_route_labels as _runtime_route_labels,
    with_node_runtime_config as _with_node_runtime_config,
    with_visit_accounting as _with_visit_accounting,
)

logger = logging.getLogger(__name__)


class NodeRegistry:
    """Registry of safe backend node implementations for compiled v2 workflows."""

    def __init__(self):
        self._nodes: Dict[str, Callable[..., Any]] = {
            WorkflowNodeType.CONTEXT_LOADER.value: self.context_loader,
            WorkflowNodeType.PLANNER.value: self.planner,
            WorkflowNodeType.ROUTER.value: self.router,
            WorkflowNodeType.RETRIEVAL_WORKER.value: self.retrieval_worker,
            WorkflowNodeType.MEMORY_WORKER.value: self.memory_worker,
            WorkflowNodeType.TIMELINE_WORKER.value: self.timeline_worker,
            WorkflowNodeType.WEB_WORKER.value: self.web_worker,
            WorkflowNodeType.EVIDENCE_EVALUATOR.value: self.evidence_evaluator,
            WorkflowNodeType.REPLANNER.value: self.replanner,
            WorkflowNodeType.DIRECT_ANSWER.value: self.direct_answer,
            WorkflowNodeType.SYNTHESIZER.value: self.synthesizer,
            WorkflowNodeType.FINALIZER.value: self.finalizer,
            WorkflowNodeType.HITL_GATE.value: self.hitl_gate,
        }

    def get(self, node_type: str) -> Callable[..., Any]:
        if node_type not in self._nodes:
            raise ValueError(f"Unknown node type: {node_type}")
        return self._nodes[node_type]

    def get_for_spec(self, node_spec: Dict[str, Any], *, route_labels: list[str] | None = None) -> Callable[..., Any]:
        node_type = str(node_spec.get("type") or "")
        node_id = str(node_spec.get("id") or node_type)
        metadata = get_node_type_metadata(node_type)
        capabilities = list(metadata.get("capabilities") or node_type_capabilities(node_type))
        node_impl = self.get(node_type)

        async def _bound_node(state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
            visit_index = _node_visit_counts(state).get(node_id, 0) + 1
            _check_visit_budget(state, node_id=node_id, node_type=node_type, visit_index=visit_index)
            runtime_config = _with_node_runtime_config(
                config,
                node_id=node_id,
                node_type=node_type,
                capabilities=capabilities,
                visit_index=visit_index,
                route_labels=route_labels,
            )
            configurable = runtime_config.get("configurable") or {}
            queue = configurable.get("studio_event_queue")
            execution_event_sink = configurable.get("execution_event_sink")
            trace_recorder = ((runtime_config.get("configurable") or {}).get("trace_recorder"))
            if trace_recorder is not None and hasattr(trace_recorder, "record_node_started"):
                trace_recorder.record_node_started(
                    node_id=node_id,
                    node_type=node_type,
                    visit_index=visit_index,
                    state=state,
                )
            if execution_event_sink is not None:
                await execution_event_sink.emit("node.started", {"node_id": node_id, "node_type": node_type, "visit_index": visit_index})
            elif queue is not None:
                await queue.put({"event": "node.started", "data": {"node_id": node_id, "node_type": node_type, "visit_index": visit_index}})
            try:
                if node_type == WorkflowNodeType.HITL_GATE.value:
                    update = await self.hitl_gate(state, runtime_config, node_id=node_id)
                else:
                    update = await node_impl(state, runtime_config)
            except Exception as exc:
                detail = None
                if trace_recorder is not None and hasattr(trace_recorder, "record_node_completed"):
                    detail = trace_recorder.record_node_completed(
                        node_id=node_id,
                        node_type=node_type,
                        visit_index=visit_index,
                        state=state,
                        update={},
                        status=NodeEventStatus.FAILED.value,
                        error=exc,
                    )
                failure_data = {"node_id": node_id, "node_type": node_type, "visit_index": visit_index, "error": str(exc), "detail": detail}
                if execution_event_sink is not None:
                    await execution_event_sink.emit("node.failed", failure_data)
                elif queue is not None:
                    await queue.put({"event": "node.failed", "data": failure_data})
                raise
            accounted_update = _with_visit_accounting(
                update,
                state,
                node_id=node_id,
                node_type=node_type,
                visit_index=visit_index,
            )
            latest_node_event = (update.get("node_events") or [{}])[-1] if isinstance(update.get("node_events"), list) and update.get("node_events") else {}
            event_name = "node.skipped" if latest_node_event.get("status") == NodeEventStatus.SKIPPED.value else "node.completed"
            detail = None
            if trace_recorder is not None and hasattr(trace_recorder, "record_node_completed"):
                detail = trace_recorder.record_node_completed(
                    node_id=node_id,
                    node_type=node_type,
                    visit_index=visit_index,
                    state=state,
                    update=accounted_update,
                    status=latest_node_event.get("status") or NodeEventStatus.COMPLETED.value,
                    event=latest_node_event,
                )
            completion_data = {
                "node_id": node_id,
                "node_type": node_type,
                "visit_index": visit_index,
                "route": update.get("route"),
                "route_reason": update.get("route_reason"),
                "evaluator_route": update.get("evaluator_route"),
                "output_preview": latest_node_event.get("output_preview"),
                "elapsed_ms": latest_node_event.get("elapsed_ms"),
                "warnings": latest_node_event.get("warnings") or latest_node_event.get("warning_codes"),
                "detail": detail,
            }
            if execution_event_sink is not None:
                await execution_event_sink.emit(event_name, completion_data)
            elif queue is not None:
                await queue.put({"event": event_name, "data": completion_data})
            return accounted_update

        return _bound_node

    async def context_loader(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            bundle = await prefetch_context(
                thread_id=state["thread_id"],
                raw_question=state["question"],
                embedding_model=state["embedding_model"],
                context_window=state.get("context_window", DEFAULT_TOKEN_BUDGET),
                use_web_search=state.get("use_web_search", False),
                use_reranker=state.get("use_reranker", True),
            )
        except Exception as exc:
            _append_failed_node_event(state, config, WorkflowNodeType.CONTEXT_LOADER.value, started, exc)
            raise
        data = {
            "status": NodeEventStatus.COMPLETED.value,
            "document_source_count": len(bundle.get("document_sources", [])),
            "web_source_count": len(bundle.get("web_sources", [])),
            "used_chat_id_count": len(bundle.get("used_chat_ids", [])),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "settings": {
                    "context_window": state.get("context_window"),
                    "use_web_search": state.get("use_web_search"),
                    "use_reranker": state.get("use_reranker"),
                },
            },
            "output_refs": _prefetch_refs(bundle),
            "output_preview": {
                "recent_history": compact_preview(bundle.get("recent_history_text")),
                "semantic_history": compact_preview(bundle.get("semantic_history_text")),
                "document_evidence": compact_preview(bundle.get("document_evidence_text")),
            },
        }
        _log_node_end(state, WorkflowNodeType.CONTEXT_LOADER.value, started, data)
        return {
            "pre_fetch_bundle": bundle,
            "document_sources": list(bundle.get("document_sources", [])),
            "web_sources": list(bundle.get("web_sources", [])),
            "used_chat_ids": list(bundle.get("used_chat_ids", [])),
            "node_events": _append_event(state, WorkflowNodeType.CONTEXT_LOADER.value, data, started=started, config=config),
        }

    async def planner(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        prompt = build_planner_prompt(state)
        response, parsed, prompt_details, retry_attempts = await invoke_json_decision_node(
            state,
            config,
            started=started,
            spec=JsonDecisionNodeSpec(
                node_name=WorkflowNodeType.PLANNER.value,
                prompt_section="Planner Node Prompt",
                system_message="You are a strict planner for a scoped RAG workflow.",
                prompt=prompt,
                failure_data={
                    "input_refs": _prefetch_refs(state.get("pre_fetch_bundle") or {}),
                    "input_preview": {
                        "question": compact_preview(state.get("question")),
                        "prefetch": compact_preview(_format_prefetch_summary(state.get("pre_fetch_bundle") or {})),
                    },
                },
            ),
            llm=llm,
            llm_retry_observer=_llm_retry_observer,
            prompt_summary=prompt_summary,
            invoke_llm_for_node=_invoke_llm_for_node,
            safe_json_object=_safe_json_object,
        )
        normalized = normalize_execution_plan(
            parsed,
            use_web_search=bool(state.get("use_web_search", False)),
            question=state.get("question"),
        )
        worker_summary = selected_and_skipped_workers(
            normalized["execution_plan"],
            WORKER_NODE_ORDER,
        )
        data = build_decision_node_event_data(
            leading_fields={
                "route": normalized["route"],
                "route_reason": normalized["route_reason"],
                "execution_plan": normalized["execution_plan"],
            },
            input_refs=_prefetch_refs(state.get("pre_fetch_bundle") or {}),
            input_preview={
                "question": compact_preview(state.get("question")),
                "prefetch": compact_preview(_format_prefetch_summary(state.get("pre_fetch_bundle") or {})),
            },
            prompt_summary=prompt_details,
            llm_result_summary={
                "parsed": bool(parsed),
                "route": normalized["route"],
                "route_reason": normalized["route_reason"],
                "execution_plan": normalized["execution_plan"],
                "clarification_option_count": len(normalized.get("clarification_options") or []),
                "normalization_notes": normalized.get("normalization_notes") or [],
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    retry_attempts=retry_attempts,
                ),
                **worker_summary,
            },
            output_refs=_prefetch_refs(state.get("pre_fetch_bundle") or {}),
            output_preview=worker_summary,
        )
        _log_node_end(state, WorkflowNodeType.PLANNER.value, started, data)
        return {
            **normalized,
            "node_events": _append_event(state, WorkflowNodeType.PLANNER.value, data, started=started, config=config),
        }

    async def router(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        prompt = build_router_prompt(state)
        configured_routes = _runtime_route_labels(config)
        if configured_routes:
            prompt += (
                "\n\nAuthoritative routes configured for this Router instance: "
                + ", ".join(configured_routes)
                + ". Return exactly one of these route values."
            )
        response, parsed, prompt_details, retry_attempts = await invoke_json_decision_node(
            state,
            config,
            started=started,
            spec=JsonDecisionNodeSpec(
                node_name=WorkflowNodeType.ROUTER.value,
                prompt_section="Router Node Prompt",
                system_message="You are a strict router for a RAG workflow.",
                prompt=prompt,
                failure_data={
                    "input_refs": _prefetch_refs(state.get("pre_fetch_bundle") or {}),
                    "input_preview": {
                        "question": compact_preview(state.get("question")),
                        "prefetch": compact_preview(_format_prefetch_summary(state.get("pre_fetch_bundle") or {})),
                    },
                },
            ),
            llm=llm,
            llm_retry_observer=_llm_retry_observer,
            prompt_summary=prompt_summary,
            invoke_llm_for_node=_invoke_llm_for_node,
            safe_json_object=_safe_json_object,
        )
        allowed_routes = set(ROUTER_ROUTES) - {RouterRoute.WEB.value}
        if state.get("use_web_search", False):
            allowed_routes.add(RouterRoute.WEB.value)
        if configured_routes:
            allowed_routes &= set(configured_routes)
        if not allowed_routes:
            raise ValueError("Router has no configured route that is enabled for this test run")
        requested_route = parsed.get("route")
        fallback_order = [
            RouterRoute.DOCUMENT.value,
            RouterRoute.DIRECT.value,
            RouterRoute.CLARIFY.value,
            RouterRoute.MEMORY.value,
            RouterRoute.TIMELINE.value,
            RouterRoute.WEB.value,
        ]
        fallback_route = next(route for route in fallback_order if route in allowed_routes)
        route = requested_route if requested_route in allowed_routes else fallback_route
        clarification_options = parsed.get("clarification_options")
        if route == RouterRoute.CLARIFY.value:
            clarification_options = _bounded_string_list(clarification_options)
            if not clarification_options:
                clarification_options = _fallback_clarification_options()
        route_reason = str(parsed.get("reason") or "")
        if requested_route != route:
            route_reason = (
                f"The model selected unavailable route {requested_route!r}; "
                f"the workflow used configured fallback {route!r}."
            )
        data = build_decision_node_event_data(
            leading_fields={
                "route": route,
                "route_reason": route_reason,
            },
            input_refs=_prefetch_refs(state.get("pre_fetch_bundle") or {}),
            input_preview={
                "question": compact_preview(state.get("question")),
                "prefetch": compact_preview(_format_prefetch_summary(state.get("pre_fetch_bundle") or {})),
            },
            prompt_summary=prompt_details,
            llm_result_summary={
                "parsed": bool(parsed),
                "route": route,
                "route_reason": route_reason,
                "clarification_option_count": len(clarification_options or []) if route == RouterRoute.CLARIFY.value else 0,
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    retry_attempts=retry_attempts,
                ),
            },
            output_refs=_prefetch_refs(state.get("pre_fetch_bundle") or {}),
        )
        _log_node_end(state, WorkflowNodeType.ROUTER.value, started, data)
        return {
            "route": route,
            "route_reason": route_reason,
            "clarification_options": clarification_options if route == RouterRoute.CLARIFY.value else None,
            "node_events": _append_event(state, WorkflowNodeType.ROUTER.value, data, started=started, config=config),
        }

    async def retrieval_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._tool_worker(WorkflowNodeType.RETRIEVAL_WORKER.value, state, config)

    async def memory_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._tool_worker(WorkflowNodeType.MEMORY_WORKER.value, state, config)

    async def timeline_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._tool_worker(WorkflowNodeType.TIMELINE_WORKER.value, state, config)

    async def web_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._tool_worker(WorkflowNodeType.WEB_WORKER.value, state, config)

    async def _tool_worker(self, node_name: str, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        spec = tool_worker_spec(node_name)
        spec = replace(spec, tool=globals().get(spec.tool_name, spec.tool))
        return await run_tool_worker(
            state,
            config,
            started=started,
            spec=spec,
            should_skip_worker=_should_skip_worker,
            skipped_worker_update=_skipped_worker_update,
            tool_config_for_node=_tool_config_for_node,
            invoke_tool_for_node=_invoke_tool_for_node,
            normalize_tool_result=normalize_tool_result,
            combine_evidence=_combine_evidence,
            evidence_text_limit=_evidence_text_limit,
            append_evidence_packet=_append_evidence_packet,
            refs_from_artifacts=refs_from_artifacts,
            state_evidence_refs=_state_evidence_refs,
            compact_refs=compact_refs,
            compact_preview=compact_preview,
            normalize_warnings=normalize_warnings,
            log_node_end=_log_node_end,
            append_event=_append_event,
            append_tool_event=_append_tool_event,
        )

    async def evidence_evaluator(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        prompt = build_evaluator_prompt(state)
        response, parsed, prompt_details, retry_attempts = await invoke_json_decision_node(
            state,
            config,
            started=started,
            spec=JsonDecisionNodeSpec(
                node_name=WorkflowNodeType.EVIDENCE_EVALUATOR.value,
                prompt_section="Evidence Evaluator Prompt",
                system_message="You are a strict evidence evaluator for a bounded RAG workflow.",
                prompt=prompt,
                failure_data={
                    "input_refs": _state_evidence_refs(state),
                    "input_preview": {
                        "question": compact_preview(state.get("question")),
                        "execution_plan": state.get("execution_plan"),
                        "evidence": compact_preview(state.get("evidence")),
                    },
                },
            ),
            llm=llm,
            llm_retry_observer=_llm_retry_observer,
            prompt_summary=prompt_summary,
            invoke_llm_for_node=_invoke_llm_for_node,
            safe_json_object=_safe_json_object,
        )
        report = normalize_evaluator_report(parsed, state)
        replan_count = _current_replan_count(state)
        replans = _replan_budget(state)
        if report["sufficient"]:
            next_route = EvaluatorRoute.ANSWER.value
            event_name = "evaluation.completed"
        elif replan_count < replans:
            next_route = EvaluatorRoute.REPLAN.value
            event_name = "replan.requested"
        else:
            next_route = EvaluatorRoute.ANSWER_BUDGET_EXHAUSTED.value
            event_name = "replan.budget_exhausted"

        evidence_update = state.get("evidence")
        if next_route == EvaluatorRoute.ANSWER_BUDGET_EXHAUSTED.value:
            gaps = "; ".join(report.get("missing_evidence") or []) or "The evaluator found unresolved evidence gaps."
            evidence_update = _combine_evidence(
                state.get("evidence"),
                (
                    "The evidence evaluator found insufficient evidence, and the replan budget is exhausted. "
                    f"Answer only from available context and explicitly state unresolved gaps: {gaps}"
                ),
                label="Evaluator warning",
                limit=_evidence_text_limit(state),
            )

        data = build_decision_node_event_data(
            leading_fields={
                "route": state.get("route"),
                "route_reason": state.get("route_reason"),
                "evaluator_route": next_route,
                "evaluator_report": report,
                "evaluation_confidence": report["confidence"],
                "evidence_gaps": report["missing_evidence"],
                "replan_count": replan_count,
                "replans": replans,
                "event_name": event_name,
            },
            input_refs=_state_evidence_refs(state),
            input_preview={
                "question": compact_preview(state.get("question")),
                "execution_plan": state.get("execution_plan"),
                "evidence": compact_preview(state.get("evidence")),
            },
            prompt_summary=prompt_details,
            llm_result_summary={
                "parsed": bool(parsed),
                "evaluator_route": next_route,
                "evaluator_report": report,
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    retry_attempts=retry_attempts,
                ),
            },
            output_refs=_state_evidence_refs({**state, "evidence": evidence_update}),
            output_preview={
                "evaluator_route": next_route,
                "evaluator_report": report,
            },
        )
        _log_node_end(state, WorkflowNodeType.EVIDENCE_EVALUATOR.value, started, data)
        return {
            "evaluator_route": next_route,
            "evaluator_report": report,
            "evidence_gaps": report["missing_evidence"],
            "evaluation_confidence": report["confidence"],
            "evidence": evidence_update,
            "node_events": _append_event(state, WorkflowNodeType.EVIDENCE_EVALUATOR.value, data, started=started, config=config),
        }

    async def replanner(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        prompt = build_replanner_prompt(state)
        response, parsed, prompt_details, retry_attempts = await invoke_json_decision_node(
            state,
            config,
            started=started,
            spec=JsonDecisionNodeSpec(
                node_name=WorkflowNodeType.REPLANNER.value,
                prompt_section="Replanner Prompt",
                system_message="You are a strict replanner for a bounded RAG workflow.",
                prompt=prompt,
                failure_data={
                    "input_refs": _state_evidence_refs(state),
                    "input_preview": {
                        "question": compact_preview(state.get("question")),
                        "current_execution_plan": state.get("execution_plan"),
                        "evaluator_report": state.get("evaluator_report"),
                    },
                },
            ),
            llm=llm,
            llm_retry_observer=_llm_retry_observer,
            prompt_summary=prompt_summary,
            invoke_llm_for_node=_invoke_llm_for_node,
            safe_json_object=_safe_json_object,
        )
        normalized = _normalize_replanner_execution_plan(
            parsed,
            use_web_search=bool(state.get("use_web_search", False)),
            allowed_tool_ids=state.get("allowed_tool_ids"),
        )
        replan_count = _current_replan_count(state) + 1
        history_item = {
            "replan_count": replan_count,
            "reason": compact_preview(normalized["reason"], limit=500),
            "execution_plan": normalized["execution_plan"],
            "evaluator_report": state.get("evaluator_report") or {},
        }
        replan_history = [
            *(state.get("replan_history") if isinstance(state.get("replan_history"), list) else []),
            history_item,
        ][-5:]
        data = build_decision_node_event_data(
            leading_fields={
                "route": state.get("route"),
                "route_reason": state.get("route_reason"),
                "execution_plan": normalized["execution_plan"],
                "replan_count": replan_count,
                "replan_reason": normalized["reason"],
                "event_name": "replan.requested" if normalized["execution_plan"] else "replan.skipped",
            },
            input_refs=_state_evidence_refs(state),
            input_preview={
                "question": compact_preview(state.get("question")),
                "current_execution_plan": state.get("execution_plan"),
                "evaluator_report": state.get("evaluator_report"),
            },
            prompt_summary=prompt_details,
            llm_result_summary={
                "parsed": bool(parsed),
                "execution_plan": normalized["execution_plan"],
                "normalization_notes": normalized.get("normalization_notes") or [],
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    retry_attempts=retry_attempts,
                ),
            },
            output_refs=_state_evidence_refs(state),
            output_preview={
                "execution_plan": normalized["execution_plan"],
                "replan_count": replan_count,
                "replan_reason": compact_preview(normalized["reason"]),
            },
        )
        _log_node_end(state, WorkflowNodeType.REPLANNER.value, started, data)
        return {
            "execution_plan": normalized["execution_plan"],
            "replan_count": replan_count,
            "replan_reason": normalized["reason"],
            "replan_history": replan_history,
            "node_events": _append_event(state, WorkflowNodeType.REPLANNER.value, data, started=started, config=config),
        }

    async def direct_answer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await direct_answer_node(state, config)

    async def synthesizer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await synthesizer_node(state, config)

    async def finalizer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await finalizer_node(state, config)

    async def hitl_gate(
        self,
        state: RouterRagState,
        config: RunnableConfig,
        *,
        node_id: str = WEB_APPROVAL_GATE_ID,
    ) -> Dict[str, Any]:
        return await hitl_gate_node(state, config, node_id=node_id)


from app.agent_workflows.compiler import WorkflowCompiler
