from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
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
from app.agent_workflows.compiler import WorkflowMaterializer
from app.agent_workflows.decision_nodes import JsonDecisionNodeSpec, invoke_json_decision_node
from app.agent_workflows.evidence import (
    append_evidence_packet as _append_evidence_packet,
    combine_evidence as _combine_evidence,
    evidence_text_limit as _evidence_text_limit,
    format_prefetch_summary as _format_prefetch_summary,
    prefetch_refs as _prefetch_refs,
    state_evidence_refs as _state_evidence_refs,
)
from app.agent_workflows.hitl_runtime import (
    WEB_APPROVAL_GATE_ID,
    hitl_gate_node,
    normalize_hitl_policy_for_thread_settings,
    with_web_approval_hitl_policy,
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
from app.agent_workflows.workers import ToolWorkerSpec, run_tool_worker
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
    refs_from_timeline,
    selected_and_skipped_workers,
)
from app.agent_workflows.state import (
    RouterRagState,
    check_visit_budget as _check_visit_budget,
    node_visit_counts as _node_visit_counts,
    with_node_runtime_config as _with_node_runtime_config,
    with_visit_accounting as _with_visit_accounting,
)

logger = logging.getLogger(__name__)


class NodeRegistry:
    """Registry of safe backend node implementations for compiled v2 workflows."""

    def __init__(self):
        self._nodes: Dict[str, Callable[..., Any]] = {
            "context_loader": self.context_loader,
            "planner": self.planner,
            "router": self.router,
            "retrieval_worker": self.retrieval_worker,
            "memory_worker": self.memory_worker,
            "timeline_worker": self.timeline_worker,
            "web_worker": self.web_worker,
            "evidence_evaluator": self.evidence_evaluator,
            "replanner": self.replanner,
            "direct_answer": self.direct_answer,
            "synthesizer": self.synthesizer,
            "finalizer": self.finalizer,
            "hitl_gate": self.hitl_gate,
        }

    def get(self, node_type: str) -> Callable[..., Any]:
        if node_type not in self._nodes:
            raise ValueError(f"Unknown node type: {node_type}")
        return self._nodes[node_type]

    def get_for_spec(self, node_spec: Dict[str, Any]) -> Callable[..., Any]:
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
            )
            if node_type == "hitl_gate":
                update = await self.hitl_gate(state, runtime_config, node_id=node_id)
            else:
                update = await node_impl(state, runtime_config)
            return _with_visit_accounting(
                update,
                state,
                node_id=node_id,
                node_type=node_type,
                visit_index=visit_index,
            )

        return _bound_node

    async def context_loader(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        try:
            bundle = await prefetch_context(
                thread_id=state["thread_id"],
                raw_question=state["question"],
                embed_model_name=state["embedding_model"],
                context_window=state.get("context_window", DEFAULT_TOKEN_BUDGET),
                use_web_search=state.get("use_web_search", False),
                use_reranker=state.get("use_reranker", True),
            )
        except Exception as exc:
            _append_failed_node_event(state, config, "context_loader", started, exc)
            raise
        data = {
            "status": "completed",
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
        _log_node_end(state, "context_loader", started, data)
        return {
            "pre_fetch_bundle": bundle,
            "document_sources": list(bundle.get("document_sources", [])),
            "web_sources": list(bundle.get("web_sources", [])),
            "used_chat_ids": list(bundle.get("used_chat_ids", [])),
            "node_events": _append_event(state, "context_loader", data, started=started, config=config),
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
                node_name="planner",
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
        data = {
            "status": "completed",
            "route": normalized["route"],
            "route_reason": normalized["route_reason"],
            "execution_plan": normalized["execution_plan"],
            "input_refs": _prefetch_refs(state.get("pre_fetch_bundle") or {}),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "prefetch": compact_preview(_format_prefetch_summary(state.get("pre_fetch_bundle") or {})),
            },
            "prompt_summary": prompt_details,
            "llm_result_summary": {
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
            "output_refs": _prefetch_refs(state.get("pre_fetch_bundle") or {}),
            "output_preview": worker_summary,
        }
        _log_node_end(state, "planner", started, data)
        return {
            **normalized,
            "node_events": _append_event(state, "planner", data, started=started, config=config),
        }

    async def router(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        prompt = build_router_prompt(state)
        response, parsed, prompt_details, retry_attempts = await invoke_json_decision_node(
            state,
            config,
            started=started,
            spec=JsonDecisionNodeSpec(
                node_name="router",
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
        allowed_routes = {"document", "memory", "timeline", "direct", "clarify"}
        if state.get("use_web_search", False):
            allowed_routes.add("web")
        route = parsed.get("route") if parsed.get("route") in allowed_routes else "document"
        clarification_options = parsed.get("clarification_options")
        if route == "clarify":
            clarification_options = _bounded_string_list(clarification_options)
            if not clarification_options:
                clarification_options = _fallback_clarification_options()
        route_reason = str(parsed.get("reason") or "")
        data = {
            "status": "completed",
            "route": route,
            "route_reason": route_reason,
            "input_refs": _prefetch_refs(state.get("pre_fetch_bundle") or {}),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "prefetch": compact_preview(_format_prefetch_summary(state.get("pre_fetch_bundle") or {})),
            },
            "prompt_summary": prompt_details,
            "llm_result_summary": {
                "parsed": bool(parsed),
                "route": route,
                "route_reason": route_reason,
                "clarification_option_count": len(clarification_options or []) if route == "clarify" else 0,
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    retry_attempts=retry_attempts,
                ),
            },
            "output_refs": _prefetch_refs(state.get("pre_fetch_bundle") or {}),
        }
        _log_node_end(state, "router", started, data)
        return {
            "route": route,
            "route_reason": route_reason,
            "clarification_options": clarification_options if route == "clarify" else None,
            "node_events": _append_event(state, "router", data, started=started, config=config),
        }

    async def retrieval_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        return await run_tool_worker(
            state,
            config,
            started=started,
            spec=ToolWorkerSpec(
                node_name="retrieval_worker",
                tool_name="search_documents",
                evidence_kind="document",
                evidence_label="Document evidence",
                tool=search_documents,
                tool_input=lambda current: {"query": current["question"], "max_results": 10},
                state_update=lambda current, _payload, artifacts, _evidence, _packets: {
                    "document_sources": [*current.get("document_sources", []), *artifacts.get("document_sources", [])],
                    "web_sources": [*current.get("web_sources", []), *artifacts.get("web_sources", [])],
                },
            ),
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

    async def memory_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        return await run_tool_worker(
            state,
            config,
            started=started,
            spec=ToolWorkerSpec(
                node_name="memory_worker",
                tool_name="search_conversation_history",
                evidence_kind="memory",
                evidence_label="Memory evidence",
                tool=search_conversation_history,
                tool_input=lambda current: {"query": current["question"], "max_results": 10},
                state_update=lambda current, _payload, artifacts, _evidence, _packets: {
                    "used_chat_ids": [*current.get("used_chat_ids", []), *artifacts.get("used_chat_ids", [])],
                },
            ),
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

    async def timeline_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        return await run_tool_worker(
            state,
            config,
            started=started,
            spec=ToolWorkerSpec(
                node_name="timeline_worker",
                tool_name="search_thread_timeline",
                evidence_kind="timeline",
                evidence_label="Timeline evidence",
                tool=search_thread_timeline,
                tool_input=lambda current: {"query": current["question"], "sources": "all", "order": "relevance", "max_results": 10},
                state_update=lambda _current, _payload, artifacts, _evidence, _packets: {
                    "timeline_event_count": len(artifacts.get("timeline_events", []) or []),
                    "timeline_refs": {"timeline_events": refs_from_timeline(artifacts.get("timeline_events"))},
                },
            ),
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

    async def web_worker(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        return await run_tool_worker(
            state,
            config,
            started=started,
            spec=ToolWorkerSpec(
                node_name="web_worker",
                tool_name="search_web",
                evidence_kind="web",
                evidence_label="Web evidence",
                tool=search_web,
                tool_input=lambda current: current["question"],
                skip_reason=lambda current: (
                    "web_search_disabled"
                    if isinstance(current.get("execution_plan"), list) and not current.get("use_web_search", False)
                    else None
                ),
                state_update=lambda current, _payload, artifacts, _evidence, _packets: {
                    "web_sources": [*current.get("web_sources", []), *artifacts.get("web_sources", [])],
                },
            ),
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
                node_name="evidence_evaluator",
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
            next_route = "answer"
            event_name = "evaluation.completed"
        elif replan_count < replans:
            next_route = "replan"
            event_name = "replan.requested"
        else:
            next_route = "answer_budget_exhausted"
            event_name = "replan.budget_exhausted"

        evidence_update = state.get("evidence")
        if next_route == "answer_budget_exhausted":
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

        data = {
            "status": "completed",
            "route": state.get("route"),
            "route_reason": state.get("route_reason"),
            "evaluator_route": next_route,
            "evaluator_report": report,
            "evaluation_confidence": report["confidence"],
            "evidence_gaps": report["missing_evidence"],
            "replan_count": replan_count,
            "replans": replans,
            "event_name": event_name,
            "input_refs": _state_evidence_refs(state),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "execution_plan": state.get("execution_plan"),
                "evidence": compact_preview(state.get("evidence")),
            },
            "prompt_summary": prompt_details,
            "llm_result_summary": {
                "parsed": bool(parsed),
                "evaluator_route": next_route,
                "evaluator_report": report,
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    retry_attempts=retry_attempts,
                ),
            },
            "output_refs": _state_evidence_refs({**state, "evidence": evidence_update}),
            "output_preview": {
                "evaluator_route": next_route,
                "evaluator_report": report,
            },
        }
        _log_node_end(state, "evidence_evaluator", started, data)
        return {
            "evaluator_route": next_route,
            "evaluator_report": report,
            "evidence_gaps": report["missing_evidence"],
            "evaluation_confidence": report["confidence"],
            "evidence": evidence_update,
            "node_events": _append_event(state, "evidence_evaluator", data, started=started, config=config),
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
                node_name="replanner",
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
        data = {
            "status": "completed",
            "route": state.get("route"),
            "route_reason": state.get("route_reason"),
            "execution_plan": normalized["execution_plan"],
            "replan_count": replan_count,
            "replan_reason": normalized["reason"],
            "event_name": "replan.requested" if normalized["execution_plan"] else "replan.skipped",
            "input_refs": _state_evidence_refs(state),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "current_execution_plan": state.get("execution_plan"),
                "evaluator_report": state.get("evaluator_report"),
            },
            "prompt_summary": prompt_details,
            "llm_result_summary": {
                "parsed": bool(parsed),
                "execution_plan": normalized["execution_plan"],
                "normalization_notes": normalized.get("normalization_notes") or [],
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    retry_attempts=retry_attempts,
                ),
            },
            "output_refs": _state_evidence_refs(state),
            "output_preview": {
                "execution_plan": normalized["execution_plan"],
                "replan_count": replan_count,
                "replan_reason": compact_preview(normalized["reason"]),
            },
        }
        _log_node_end(state, "replanner", started, data)
        return {
            "execution_plan": normalized["execution_plan"],
            "replan_count": replan_count,
            "replan_reason": normalized["reason"],
            "replan_history": replan_history,
            "node_events": _append_event(state, "replanner", data, started=started, config=config),
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

class WorkflowCompiler(WorkflowMaterializer):
    """Compile validated v2 workflow specs into LangGraph StateGraph instances."""

    def __init__(self, registry: Optional[NodeRegistry] = None):
        self.registry = registry or NodeRegistry()

    def compile(
        self,
        spec: Dict[str, Any],
        *,
        checkpointer: Any = None,
    ):
        from app.agent_workflows.validator import WorkflowValidator

        graph_spec = ((spec.get("config") or {}).get("graph") or {}) if isinstance(spec, dict) else {}
        if not graph_spec.get("hitl_compiled"):
            WorkflowValidator().validate(spec)
            spec = self.materialize_spec(spec)
            graph_spec = (spec.get("config") or {}).get("graph") or {}
        workflow = StateGraph(RouterRagState)
        node_types: Dict[str, str] = {}
        for node in graph_spec.get("nodes", []):
            node_types[node["id"]] = node["type"]
            workflow.add_node(node["id"], self.registry.get_for_spec(node))

        for edge in graph_spec.get("edges", []):
            source = edge.get("from")
            target = edge.get("to")
            if edge.get("conditional"):
                route_fn = _route_function_for_edge(
                    edge,
                    source=str(source),
                    node_types=node_types,
                )
                routes = {
                    key: END if value == "END" else value
                    for key, value in dict(edge["routes"]).items()
                }
                workflow.add_conditional_edges(source, route_fn, routes)
                continue
            source_ref = START if source == "START" else source
            target_ref = END if target == "END" else target
            workflow.add_edge(source_ref, target_ref)

        return workflow.compile(checkpointer=checkpointer)
