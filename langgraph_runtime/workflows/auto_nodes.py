"""Auto orchestrator delegation nodes."""

from __future__ import annotations

import re
import time
from typing import Any, Dict, Mapping

from langchain_core.runnables import RunnableConfig

from langgraph_runtime.control_plane_client import delegate_sync
from langgraph_runtime.workflows.corrective_nodes import DOCUMENT_SCOPE_DECLINE_ANSWER
from langgraph_runtime.workflows.enums import AutoOrchestratorRoute, NodeEventStatus
from langgraph_runtime.workflows.runtime_invocation import append_event, log_node_end
from langgraph_runtime.workflows.state import RouterRagState
from langgraph_runtime.workflows.trace import compact_preview

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
_WORLD_FACT_PATTERN = re.compile(
    r"\b(population of|capital of|weather in|how many people live)\b",
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

    Out-of-scope world-fact questions with attached documents but no prefetch evidence
    may decline only when Auto would route to the simple router delegate.
  Deep Research and multi-document orchestration paths are never pre-empted.
    """

    if workflow_id in _DELEGATE_SKIP_SCOPE_DECLINE:
        return None
    if not workflow_id:
        return DOCUMENT_SCOPE_DECLINE_ANSWER
    bundle = state.get("pre_fetch_bundle") or {}
    attached = bool(state.get("document_sources"))
    if isinstance(bundle, dict):
        documents = bundle.get("documents")
        if isinstance(documents, list) and documents:
            attached = True
        document_sources = bundle.get("document_sources")
        if isinstance(document_sources, list) and document_sources:
            attached = True
    if not attached:
        return None
    evidence = ""
    if isinstance(bundle, dict):
        for key in ("document_evidence_text", "web_search_evidence_text", "web_evidence_text"):
            evidence = str(bundle.get(key) or "").strip()
            if evidence:
                break
    question = str(state.get("question") or "")
    if _WORLD_FACT_PATTERN.search(question):
        return DOCUMENT_SCOPE_DECLINE_ANSWER
    if evidence:
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


def _choose_delegate_workflow(state: Mapping[str, Any], *, catalog_ids: set[str]) -> tuple[str, str]:
    question = str(state.get("question") or "")
    if DEEP_RESEARCH_WORKFLOW_ID in catalog_ids and _DEEP_RESEARCH_PATTERN.search(question):
        return DEEP_RESEARCH_WORKFLOW_ID, "Deep Research fits this open-ended research request."
    doc_count = _document_count(state)
    if doc_count >= 2 or _COMPLEX_PATTERN.search(question):
        if ORCHESTRATOR_WORKFLOW_ID in catalog_ids:
            return ORCHESTRATOR_WORKFLOW_ID, "Run the parallel orchestrator across the attached documents."
        if PLAN_EXECUTE_WORKFLOW_ID in catalog_ids:
            return PLAN_EXECUTE_WORKFLOW_ID, "Run the planner workflow for a multi-step document answer."
    if ROUTER_WORKFLOW_ID in catalog_ids:
        return ROUTER_WORKFLOW_ID, "Run the router workflow for a direct document question."
    for workflow_id in sorted(catalog_ids):
        if workflow_id not in {AUTO_WORKFLOW_ID, DEEP_RESEARCH_WORKFLOW_ID}:
            return workflow_id, f"Run workflow {workflow_id}."
    return "", "No eligible delegate workflow is available."


def _auto_policy(state: Mapping[str, Any]) -> Dict[str, Any]:
    # Resolved spec stores auto_policy on config; execution state may mirror it later.
    return {}


async def auto_orchestrator_node(state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    catalog_ids = _catalog_ids(state)
    world_fact_decline = auto_scope_decline(state, workflow_id=ROUTER_WORKFLOW_ID)
    if world_fact_decline:
        data = {
            "status": NodeEventStatus.COMPLETED.value,
            "auto_orchestrator_route": AutoOrchestratorRoute.DECLINE.value,
            "delegate_workflow_id": "",
            "output_preview": {"final_answer": compact_preview(world_fact_decline)},
        }
        log_node_end(state, "auto_orchestrator", started, data)
        return {
            "auto_orchestrator_route": AutoOrchestratorRoute.DECLINE.value,
            "final_answer": world_fact_decline,
            "document_scope_declined": True,
            "node_events": append_event(state, "auto_orchestrator", data, started=started, config=config),
        }

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
        require_approval = workflow_id == DEEP_RESEARCH_WORKFLOW_ID or (
            _COMPLEX_PATTERN.search(str(state.get("question") or "")) or _document_count(state) >= 2
        )
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
    if not child_workflow_id or not parent_run_id or not thread_id:
        raise ValueError("workflow_delegate requires delegate_workflow_id, agent_run_id, and thread_id")

    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), dict) else {}
    response = await delegate_sync(
        parent_run_id=parent_run_id,
        thread_id=thread_id,
        child_workflow_id=child_workflow_id,
        question=question,
        pre_fetch_bundle=bundle,
        config=config,
        llm_model=state.get("llm_model"),
        use_web_search=bool(state.get("use_web_search")),
        use_reranker=state.get("use_reranker"),
        system_role=str(state.get("system_role") or ""),
        custom_instructions=str(state.get("custom_instructions") or ""),
        tool_instructions=state.get("tool_instructions") if isinstance(state.get("tool_instructions"), dict) else {},
    )
    child_run_id = str(response.get("child_run_id") or "")
    child_final_answer = str(response.get("final_answer") or response.get("message") or "")
    agent_task_id = response.get("agent_task_id")
    data = {
        "status": NodeEventStatus.COMPLETED.value,
        "child_run_id": child_run_id,
        "child_workflow_id": child_workflow_id,
        "delegate_child_run_id": child_run_id,
        "delegate_child_workflow_id": child_workflow_id,
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
        "node_events": append_event(state, "workflow_delegate", data, started=started, config=config),
    }
    if agent_task_id:
        update["agent_task_id"] = agent_task_id
    return update


async def delegate_synthesizer_node(state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
    started = time.perf_counter()
    answer = str(state.get("child_final_answer") or state.get("final_answer") or "").strip()
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
