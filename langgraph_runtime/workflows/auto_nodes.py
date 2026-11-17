"""Auto orchestrator delegation nodes."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Mapping

from langchain_core.runnables import RunnableConfig

from langgraph_runtime.control_plane_client import delegate_sync
from langgraph_runtime.workflows.document_scope import (
    ATTACHMENT_IRRELEVANT_RETRIEVAL_SCORE,
    DOCUMENT_SCOPE_DECLINE_ANSWER,
    attachment_retrieval_scope_decline,
    prefetch_document_retrieval_scores,
    thread_has_attached_documents,
)
from langgraph_runtime.workflows.external_research import attachment_document_scope_bypass
from langgraph_runtime.workflows.enums import AutoOrchestratorRoute, NodeEventStatus
from langgraph_runtime.workflows.auto_limit_errors import AutoLimitExceeded
from langgraph_runtime.workflows.runtime_invocation import append_event, log_node_end
from runtime_protocol.auto_limits import AUTO_LIMIT_MAX_DELEGATES_PER_TURN
from langgraph_runtime.workflows.state import RouterRagState
from langgraph_runtime.workflows.trace import compact_preview
from langgraph_runtime.workflows.clarification_terminal import (
    clarification_options_from_mapping,
    clarification_terminal_active,
    clarification_terminal_state_update,
    is_clarification_terminal_status,
)

AUTO_WORKFLOW_ID = "auto_rag_agent"
ROUTER_WORKFLOW_ID = "router_rag_agent"
ORCHESTRATOR_WORKFLOW_ID = "orchestrator_worker_rag_agent"
DEEP_RESEARCH_WORKFLOW_ID = "deep_research_agent"
PLAN_EXECUTE_WORKFLOW_ID = "plan_execute_rag_agent"

_DEEP_RESEARCH_PATTERN = re.compile(
    r"\b(deep research|comprehensive report|literature review|multi-?hour research)\b",
    re.IGNORECASE,
)
_COMPLEX_PATTERN = re.compile(
    r"\b(compare|contrast|versus|vs\.?|across|both papers|two papers|multiple documents)\b",
    re.IGNORECASE,
)


def _catalog_ids(state: Mapping[str, Any]) -> set[str]:
    catalog = state.get("available_delegate_workflows") or []
    if not isinstance(catalog, list):
        return set()
    return {
        str(item.get("workflow_id") or "").strip()
        for item in catalog
        if isinstance(item, dict) and str(item.get("workflow_id") or "").strip()
    }


_DELEGATE_SKIP_SCOPE_DECLINE = frozenset(
    {
        DEEP_RESEARCH_WORKFLOW_ID,
        ORCHESTRATOR_WORKFLOW_ID,
        PLAN_EXECUTE_WORKFLOW_ID,
    }
)


def auto_scope_decline(state: Mapping[str, Any], *, workflow_id: str = "") -> str | None:
    """Canonical document-scope decline for Auto after delegate choice.

    When documents are attached but pre-fetch produced no document (or web) evidence,
    decline only on the simple router path. Deep Research and orchestration delegates
    are never pre-empted here.
    """

    if attachment_document_scope_bypass(state):
        return None
    if workflow_id in _DELEGATE_SKIP_SCOPE_DECLINE:
        return None
    if not workflow_id:
        return DOCUMENT_SCOPE_DECLINE_ANSWER
    if workflow_id != ROUTER_WORKFLOW_ID:
        return None
    if not thread_has_attached_documents(state):
        return None
    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), dict) else {}
    for key in ("web_search_evidence_text", "web_evidence_text"):
        if str(bundle.get(key) or "").strip():
            return None
    if str(bundle.get("document_evidence_text") or "").strip():
        scores = prefetch_document_retrieval_scores(state)
        if not scores or max(scores) >= ATTACHMENT_IRRELEVANT_RETRIEVAL_SCORE:
            return None
    score_decline = attachment_retrieval_scope_decline(state)
    if score_decline:
        return None
    if not str(bundle.get("document_evidence_text") or "").strip():
        return None
    return None


def _document_count(state: Mapping[str, Any]) -> int:
    bundle = state.get("pre_fetch_bundle") or {}
    counts: list[int] = []
    if isinstance(bundle, dict):
        documents = bundle.get("documents")
        if isinstance(documents, list):
            counts.append(len(documents))
        document_sources = bundle.get("document_sources")
        if isinstance(document_sources, list):
            counts.append(len(document_sources))
    sources = state.get("document_sources")
    if isinstance(sources, list):
        counts.append(len(sources))
    return max(counts) if counts else 0


def _require_plan_approval(state: Mapping[str, Any], *, workflow_id: str) -> bool:
    policy = _auto_policy(state)
    if policy.get("require_plan_approval_all"):
        return bool(workflow_id)
    if policy.get("require_plan_approval_on_complex"):
        if workflow_id in {
            DEEP_RESEARCH_WORKFLOW_ID,
            ORCHESTRATOR_WORKFLOW_ID,
            PLAN_EXECUTE_WORKFLOW_ID,
        }:
            return True
    return workflow_id == DEEP_RESEARCH_WORKFLOW_ID


def _choose_delegate_workflow(state: Mapping[str, Any], *, catalog_ids: set[str]) -> tuple[str, str]:
    question = str(state.get("question") or "")
    if DEEP_RESEARCH_WORKFLOW_ID in catalog_ids and _DEEP_RESEARCH_PATTERN.search(question):
        return DEEP_RESEARCH_WORKFLOW_ID, "Deep Research fits this open-ended research request."
    if _COMPLEX_PATTERN.search(question):
        if ORCHESTRATOR_WORKFLOW_ID in catalog_ids:
            return ORCHESTRATOR_WORKFLOW_ID, "Run the parallel orchestrator across the attached documents."
        if PLAN_EXECUTE_WORKFLOW_ID in catalog_ids:
            return PLAN_EXECUTE_WORKFLOW_ID, "Run the planner workflow for a multi-step document answer."
    if ROUTER_WORKFLOW_ID in catalog_ids:
        return ROUTER_WORKFLOW_ID, "Run the router workflow for this question."
    for workflow_id in sorted(catalog_ids):
        if workflow_id not in {AUTO_WORKFLOW_ID, DEEP_RESEARCH_WORKFLOW_ID}:
            return workflow_id, f"Run workflow {workflow_id}."
    return "", "No eligible delegate workflow is available."


def _auto_policy(state: Mapping[str, Any]) -> Dict[str, Any]:
    policy = state.get("auto_policy")
    return dict(policy) if isinstance(policy, dict) else {}


def _trace_recorder(config: RunnableConfig) -> Any:
    return ((config or {}).get("configurable") or {}).get("trace_recorder")


def _delegation_detail(
    *,
    child_run_id: str,
    child_workflow_id: str,
    question: str,
    bundle: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "child_run_id": child_run_id,
        "child_workflow_id": child_workflow_id,
        "question_chars": len(question),
        "prefetch_seeded": bool(bundle),
    }


async def auto_orchestrator_node(state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    catalog_ids = _catalog_ids(state)
    workflow_id, rationale = _choose_delegate_workflow(state, catalog_ids=catalog_ids)
    decline = auto_scope_decline(state, workflow_id=workflow_id)
    if decline:
        data = {
            "status": NodeEventStatus.COMPLETED.value,
            "auto_orchestrator_route": AutoOrchestratorRoute.DECLINE.value,
            "delegate_workflow_id": "",
            "output_preview": {"final_answer": compact_preview(decline)},
        }
        log_node_end(state, "auto_orchestrator", started, data)
        return {
            "auto_orchestrator_route": AutoOrchestratorRoute.DECLINE.value,
            "final_answer": decline,
            "document_scope_declined": True,
            "node_events": append_event(state, "auto_orchestrator", data, started=started, config=config),
        }

    if not workflow_id:
        route = AutoOrchestratorRoute.DECLINE.value
    else:
        require_approval = _require_plan_approval(state, workflow_id=workflow_id)
        route = AutoOrchestratorRoute.PLAN_APPROVAL.value if require_approval else AutoOrchestratorRoute.DELEGATE.value

    plan_summary = f"Delegate to `{workflow_id}`: {rationale}" if workflow_id else rationale
    data = {
        "status": NodeEventStatus.COMPLETED.value,
        "auto_orchestrator_route": route,
        "delegate_workflow_id": workflow_id,
        "delegate_plan_summary": compact_preview(plan_summary),
        "output_preview": {"route": route, "workflow_id": workflow_id},
    }
    log_node_end(state, "auto_orchestrator", started, data)
    return {
        "auto_orchestrator_route": route,
        "delegate_workflow_id": workflow_id,
        "delegate_plan_summary": plan_summary,
        "delegate_requires_approval": route == AutoOrchestratorRoute.PLAN_APPROVAL.value,
        "hitl_plan_preview": plan_summary,
        "node_events": append_event(state, "auto_orchestrator", data, started=started, config=config),
    }


async def workflow_delegate_node(state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    child_workflow_id = str(state.get("delegate_workflow_id") or "").strip()
    parent_run_id = str(state.get("agent_run_id") or "").strip()
    thread_id = str(state.get("thread_id") or "").strip()
    question = str(state.get("question") or "")
    policy = _auto_policy(state)
    try:
        delegate_cap = max(1, int(policy.get("max_delegates_per_turn", 1)))
    except (TypeError, ValueError):
        delegate_cap = 1
    invocation_count = int(state.get("delegate_invocation_count") or 0)
    if invocation_count >= delegate_cap:
        raise AutoLimitExceeded(
            AUTO_LIMIT_MAX_DELEGATES_PER_TURN,
            limit=delegate_cap,
            state=state,
        )
    if not child_workflow_id or not parent_run_id or not thread_id:
        raise ValueError("workflow_delegate requires delegate_workflow_id, agent_run_id, and thread_id")

    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), dict) else {}
    trace_recorder = _trace_recorder(config)
    if trace_recorder is not None and hasattr(trace_recorder, "record_runtime_event"):
        trace_recorder.record_runtime_event(
            "delegation.started",
            attributes={
                "child_workflow_id": child_workflow_id,
                "parent_run_id": parent_run_id,
                "question_chars": len(question),
            },
        )
    response = await delegate_sync(
        parent_run_id=parent_run_id,
        thread_id=thread_id,
        child_workflow_id=child_workflow_id,
        question=question,
        pre_fetch_bundle=bundle,
        config=config,
        llm_model=state.get("llm_model"),
        use_web_search=bool(state.get("use_web_search")),
        use_browser_access=bool(state.get("use_browser_access")),
        browser_access_mode=str(state.get("browser_access_mode") or "off"),
        use_reranker=state.get("use_reranker"),
        system_role=str(state.get("system_role") or ""),
        custom_instructions=str(state.get("custom_instructions") or ""),
        tool_instructions=state.get("tool_instructions") if isinstance(state.get("tool_instructions"), dict) else {},
        context_window=state.get("context_window"),
        context_budget=state.get("context_budget") if isinstance(state.get("context_budget"), dict) else None,
    )
    child_run_id = str(response.get("child_run_id") or "")
    child_result = response.get("result") if isinstance(response.get("result"), dict) else {}
    child_status = response.get("status") or child_result.get("status")
    child_options = clarification_options_from_mapping({**child_result, "clarification_options": response.get("clarification_options") or child_result.get("clarification_options")})
    delegation = _delegation_detail(
        child_run_id=child_run_id,
        child_workflow_id=child_workflow_id,
        question=question,
        bundle=bundle,
    )
    if trace_recorder is not None and hasattr(trace_recorder, "record_runtime_event") and child_run_id:
        trace_recorder.record_runtime_event(
            "delegation.completed",
            attributes={
                **delegation,
                "parent_run_id": parent_run_id,
                "child_status": child_status,
            },
        )
    if clarification_terminal_active({**child_result, "status": child_status, "clarification_options": child_options}):
        data = {
            "status": NodeEventStatus.COMPLETED.value,
            "child_run_id": child_run_id,
            "child_workflow_id": child_workflow_id,
            "delegation": delegation,
            "output_preview": {"clarification_options": child_options},
            "llm_result_summary": {"clarification_option_count": len(child_options)},
        }
        log_node_end(state, "workflow_delegate", started, data)
        return {
            **clarification_terminal_state_update(child_options),
            "child_run_id": child_run_id,
            "child_workflow_id": child_workflow_id,
            "delegate_child_run_id": child_run_id,
            "delegate_child_workflow_id": child_workflow_id,
            "delegate_child_output": response,
            "node_events": append_event(state, "workflow_delegate", data, started=started, config=config),
        }

    child_final_answer = str(response.get("final_answer") or response.get("message") or child_result.get("answer") or "")
    if is_clarification_terminal_status(child_status) and child_final_answer:
        # Legacy child runs may still return prose; drop it so the card can own UX.
        child_final_answer = ""
    agent_task_id = response.get("agent_task_id")
    data = {
        "status": NodeEventStatus.COMPLETED.value,
        "child_run_id": child_run_id,
        "child_workflow_id": child_workflow_id,
        "delegate_child_run_id": child_run_id,
        "delegate_child_workflow_id": child_workflow_id,
        "delegation": delegation,
        "agent_task_id": agent_task_id,
        "output_preview": {"answer": compact_preview(child_final_answer)},
    }
    log_node_end(state, "workflow_delegate", started, data)
    update: Dict[str, Any] = {
        "child_run_id": child_run_id,
        "child_workflow_id": child_workflow_id,
        "delegate_child_run_id": child_run_id,
        "delegate_child_workflow_id": child_workflow_id,
        "child_final_answer": child_final_answer,
        "delegate_child_output": response,
        "delegate_invocation_count": invocation_count + 1,
        "node_events": append_event(state, "workflow_delegate", data, started=started, config=config),
    }
    if agent_task_id:
        update["agent_task_id"] = agent_task_id
    return update


async def delegate_synthesizer_node(state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    if state.get("clarification_options"):
        options = list(state.get("clarification_options") or [])
        data = {
            "status": NodeEventStatus.COMPLETED.value,
            "child_run_id": state.get("child_run_id"),
            "child_workflow_id": state.get("child_workflow_id"),
            "output_preview": {"clarification_options": options},
            "llm_result_summary": {"clarification_option_count": len(options)},
        }
        log_node_end(state, "delegate_synthesizer", started, data)
        return {
            **clarification_terminal_state_update(options),
            "debug_trace": state.get("debug_trace"),
            "node_events": append_event(state, "delegate_synthesizer", data, started=started, config=config),
        }
    child_output = state.get("delegate_child_output") if isinstance(state.get("delegate_child_output"), dict) else {}
    notice = str(state.get("auto_limit_notice") or "").strip()
    answer = str(state.get("child_final_answer") or state.get("final_answer") or "").strip()
    if notice:
        answer = notice if not answer else f"{notice}\n\n{answer}"
    if (
        not answer
        and str(child_output.get("status") or "") == "awaiting_human"
        and child_output.get("pending_interrupt")
    ):
        answer = "Waiting for your approval to continue the delegated workflow."
    if not answer and state.get("agent_task_id"):
        task_id = str(state.get("agent_task_id"))
        answer = f"Deep Research task started. Track progress in task `{task_id}`."
    data = {
        "status": NodeEventStatus.COMPLETED.value,
        "child_run_id": state.get("child_run_id"),
        "child_workflow_id": state.get("child_workflow_id"),
        "output_preview": {"final_answer": compact_preview(answer)},
    }
    log_node_end(state, "delegate_synthesizer", started, data)
    debug_trace = dict(state.get("debug_trace") or {}) if isinstance(state.get("debug_trace"), dict) else {}
    technical = dict(debug_trace.get("technical_details") or {})
    technical.update(
        {
            "child_run_id": state.get("child_run_id"),
            "child_workflow_id": state.get("child_workflow_id"),
        }
    )
    debug_trace["technical_details"] = technical
    return {
        "final_answer": answer,
        "debug_trace": debug_trace,
        "node_events": append_event(state, "delegate_synthesizer", data, started=started, config=config),
    }
