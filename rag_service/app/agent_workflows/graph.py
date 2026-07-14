from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.agent.reasoning import normalize_ai_response
from app.agent.tool_contract import normalize_tool_result
from app.agent.tool_registry import get_tool_contract_id, validate_tool_call_allowed
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET, get_llm
from app.models.retry import invoke_with_retry
from app.agent.external_research_tools import search_web
from app.rag.agent_tools import search_conversation_history, search_documents, search_thread_timeline
from app.rag.chat_service import prefetch_context
from app.agent_workflows.prompting import (
    build_evaluator_prompt,
    build_final_answer_messages,
    build_planner_prompt,
    build_replanner_prompt,
    build_router_prompt,
)
from app.agent_workflows.compiler import TemplateMaterializer
from app.agent_workflows.decision_nodes import JsonDecisionNodeSpec, invoke_json_decision_node
from app.agent_workflows.evidence import (
    append_evidence_packet as _append_evidence_packet,
    combine_evidence as _combine_evidence,
    evidence_text_limit as _evidence_text_limit,
    final_context_from_state as _final_context_from_state,
    format_prefetch_summary as _format_prefetch_summary,
    prefetch_refs as _prefetch_refs,
    state_evidence_refs as _state_evidence_refs,
)
from app.agent_workflows.events import append_node_event, append_tool_event
from app.agent_workflows.planning import (
    WORKER_NODE_ORDER,
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
    node_type_default_max_visits,
    node_type_max_visits,
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
from app.time_utils import iso_utc_z, utc_now


RouterRoute = Literal["document", "memory", "timeline", "web", "direct", "clarify"]

logger = logging.getLogger(__name__)
FINAL_REVIEW_GATE_ID = "human_review_gate"
WEB_APPROVAL_GATE_ID = "web_approval_gate"
NODE_RUNTIME_CONFIG_KEY = "agent_workflow_node_runtime"


def _node_runtime(config: Optional[RunnableConfig]) -> Dict[str, Any]:
    configurable = ((config or {}).get("configurable") or {})
    runtime = configurable.get(NODE_RUNTIME_CONFIG_KEY)
    return runtime if isinstance(runtime, dict) else {}


def _runtime_node_id(config: Optional[RunnableConfig], fallback: str) -> str:
    runtime = _node_runtime(config)
    node_id = runtime.get("node_id")
    return str(node_id) if isinstance(node_id, str) and node_id else fallback


def _runtime_node_type(config: Optional[RunnableConfig], fallback: str) -> str:
    runtime = _node_runtime(config)
    node_type = runtime.get("node_type")
    return str(node_type) if isinstance(node_type, str) and node_type else fallback


def _runtime_node_capabilities(config: Optional[RunnableConfig]) -> List[str]:
    runtime = _node_runtime(config)
    capabilities = runtime.get("capabilities")
    return [str(item) for item in capabilities] if isinstance(capabilities, list) else []


def _runtime_visit_index(config: Optional[RunnableConfig]) -> Optional[int]:
    runtime = _node_runtime(config)
    try:
        value = int(runtime.get("visit_index"))
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def _with_node_runtime_config(
    config: Optional[RunnableConfig],
    *,
    node_id: str,
    node_type: str,
    capabilities: List[str],
    visit_index: int,
) -> RunnableConfig:
    updated = dict(config or {})
    configurable = dict(updated.get("configurable") or {})
    configurable[NODE_RUNTIME_CONFIG_KEY] = {
        "node_id": node_id,
        "node_type": node_type,
        "capabilities": list(capabilities),
        "visit_index": visit_index,
    }
    updated["configurable"] = configurable
    metadata = dict(updated.get("metadata") or {})
    metadata.update(
        {
            "node_id": node_id,
            "node_type": node_type,
            "node_capabilities": list(capabilities),
            "node_visit_index": visit_index,
        }
    )
    updated["metadata"] = metadata
    return updated


def _loop_policy(state: RouterRagState) -> Dict[str, Any]:
    policy = state.get("loop_policy")
    return policy if isinstance(policy, dict) else {}


def _node_visit_counts(state: RouterRagState) -> Dict[str, int]:
    counts = state.get("node_visit_counts")
    if not isinstance(counts, dict):
        return {}
    normalized: Dict[str, int] = {}
    for key, value in counts.items():
        if not isinstance(key, str):
            continue
        try:
            normalized[key] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _node_visit_sequence(state: RouterRagState) -> List[Dict[str, Any]]:
    sequence = state.get("node_visit_sequence")
    return [item for item in sequence if isinstance(item, dict)] if isinstance(sequence, list) else []


def _node_visit_limit(state: RouterRagState, *, node_id: str, node_type: str) -> Optional[int]:
    policy = _loop_policy(state)
    if not policy:
        return None
    node_limits = policy.get("node_visit_limits") if isinstance(policy.get("node_visit_limits"), dict) else {}
    if node_id in node_limits:
        try:
            return max(1, int(node_limits[node_id]))
        except (TypeError, ValueError):
            return 1
    try:
        default_limit = int(policy.get("default_max_node_visits", node_type_default_max_visits(node_type)))
    except (TypeError, ValueError):
        default_limit = node_type_default_max_visits(node_type)
    return max(1, min(default_limit, node_type_max_visits(node_type)))


def _total_visit_limit(state: RouterRagState) -> Optional[int]:
    policy = _loop_policy(state)
    if not policy:
        return None
    try:
        value = int(policy.get("max_total_visits"))
    except (TypeError, ValueError):
        return None
    return max(1, value)


def _check_visit_budget(state: RouterRagState, *, node_id: str, node_type: str, visit_index: int) -> None:
    limit = _node_visit_limit(state, node_id=node_id, node_type=node_type)
    if limit is not None and visit_index > limit:
        raise ValueError(f"Node {node_id} exceeded visit limit {limit}")
    total_limit = _total_visit_limit(state)
    if total_limit is not None and len(_node_visit_sequence(state)) + 1 > total_limit:
        raise ValueError(f"Graph exceeded total visit limit {total_limit}")


def _with_visit_accounting(
    update: Dict[str, Any],
    state: RouterRagState,
    *,
    node_id: str,
    node_type: str,
    visit_index: int,
) -> Dict[str, Any]:
    counts = _node_visit_counts(state)
    counts[node_id] = max(counts.get(node_id, 0), visit_index)
    sequence = [
        *_node_visit_sequence(state),
        {"node": node_id, "node_type": node_type, "visit_index": visit_index},
    ]
    return {
        **update,
        "node_visit_counts": counts,
        "node_visit_sequence": sequence,
    }


def _hitl_interrupt_counts(state: RouterRagState) -> Dict[str, int]:
    counts = state.get("hitl_interrupt_counts")
    if not isinstance(counts, dict):
        return {}
    normalized: Dict[str, int] = {}
    for key, value in counts.items():
        if not isinstance(key, str):
            continue
        try:
            normalized[key] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _hitl_interrupt_limit(policy: Dict[str, Any], gate_policy: Dict[str, Any]) -> Optional[int]:
    raw = gate_policy.get("max_interrupts_per_run", policy.get("max_interrupts_per_run"))
    if raw in (None, ""):
        return None
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def _hitl_interrupt_count_key(gate_id: str, visit_index: Optional[int]) -> str:
    if isinstance(visit_index, int) and visit_index >= 1:
        return f"{gate_id}:visit:{visit_index}"
    return gate_id


def _hitl_visit_interrupt_count(counts: Dict[str, int], *, gate_id: str, visit_index: Optional[int]) -> int:
    visit_key = _hitl_interrupt_count_key(gate_id, visit_index)
    has_visit_counts = any(key.startswith(f"{gate_id}:visit:") for key in counts)
    if visit_key != gate_id and has_visit_counts:
        return counts.get(visit_key, 0)
    return counts.get(gate_id, 0)


class RouterRagState(TypedDict, total=False):
    agent_run_id: Optional[str]
    thread_id: str
    question: str
    llm_model: str
    embedding_model: str
    context_window: int
    use_web_search: bool
    use_reranker: bool
    system_role: str
    tool_instructions: Dict[str, str]
    custom_instructions: str
    client_timezone: Optional[str]
    client_locale: Optional[str]
    client_now_iso: Optional[str]
    pre_fetch_bundle: Dict[str, Any]
    route: RouterRoute
    route_reason: str
    clarification_options: Optional[List[str]]
    evidence: str
    document_sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    used_chat_ids: List[str]
    final_answer: str
    reasoning: str
    reasoning_available: bool
    reasoning_format: str
    human_review_decision: Dict[str, Any]
    hitl_policy: Dict[str, Any]
    hitl_decisions: List[Dict[str, Any]]
    hitl_gate_route: str
    hitl_gate_routes: Dict[str, Any]
    hitl_selected_options: Dict[str, List[str]]
    skipped_nodes: List[str]
    node_events: List[Dict[str, Any]]
    tool_events: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    allowed_tool_ids: List[str]
    pattern_type: str
    loop_policy: Dict[str, Any]
    node_visit_counts: Dict[str, int]
    node_visit_sequence: List[Dict[str, Any]]
    evidence_packets: List[Dict[str, Any]]
    hitl_interrupt_counts: Dict[str, int]
    execution_plan: List[str]
    replans: int
    replan_count: int
    replan_reason: str
    replan_history: List[Dict[str, Any]]
    evaluator_report: Dict[str, Any]
    evidence_gaps: List[str]
    evaluation_confidence: float
    evaluator_route: str


def _append_event(
    state: RouterRagState,
    node: str,
    data: Optional[Dict[str, Any]] = None,
    *,
    started: Optional[float] = None,
    config: Optional[RunnableConfig] = None,
) -> List[Dict[str, Any]]:
    return append_node_event(
        state,
        node,
        data,
        started=started,
        config=config,
        runtime_node_id=_runtime_node_id,
        runtime_node_type=_runtime_node_type,
        runtime_visit_index=_runtime_visit_index,
        utc_now=utc_now,
        iso_utc_z=iso_utc_z,
    )


def _append_tool_event(
    state: RouterRagState,
    payload: Dict[str, Any],
    *,
    tool_input: Any = None,
    config: Optional[RunnableConfig] = None,
) -> List[Dict[str, Any]]:
    return append_tool_event(
        state,
        payload,
        tool_input=tool_input,
        config=config,
        runtime_node_type=_runtime_node_type,
        runtime_visit_index=_runtime_visit_index,
    )


def _error_summary(exc: Exception, *, code: str) -> Dict[str, Any]:
    return {
        "code": code,
        "type": type(exc).__name__,
        "message": compact_preview(str(exc), limit=700),
        "raw_message": compact_preview(str(exc), limit=700),
        "retryable": True,
    }


def _append_failed_node_event(
    state: RouterRagState,
    config: RunnableConfig,
    node: str,
    started: float,
    exc: Exception,
    *,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "status": "failed",
        "error": _error_summary(exc, code=f"{node}_failed"),
        "input_preview": {"question": compact_preview(state.get("question"))},
        **(data or {}),
    }
    _append_event(state, node, payload, started=started, config=config)


async def _invoke_llm_for_node(
    func: Callable[..., Any],
    messages: List[Any],
    *,
    state: RouterRagState,
    config: RunnableConfig,
    node: str,
    started: float,
    retry_observer: Callable[[Dict[str, Any]], None],
    retry_attempts: List[Dict[str, Any]],
    model_name: Optional[str],
    failure_data: Optional[Dict[str, Any]] = None,
) -> Any:
    try:
        return await invoke_with_retry(func, messages, retry_observer=retry_observer)
    except Exception as exc:
        llm_failure = {
            "llm_result_summary": {
                "llm": {
                    "model_name": model_name,
                    "retry_count": len(retry_attempts),
                    "retry_attempts": retry_attempts,
                }
            }
        }
        _append_failed_node_event(
            state,
            config,
            node,
            started,
            exc,
            data={**(failure_data or {}), **llm_failure},
        )
        raise


async def _invoke_tool_for_node(
    tool: Any,
    tool_input: Any,
    *,
    state: RouterRagState,
    config: RunnableConfig,
    node: str,
    started: float,
) -> Any:
    try:
        return await tool.ainvoke(tool_input, config=config)
    except Exception as exc:
        _append_failed_node_event(
            state,
            config,
            node,
            started,
            exc,
            data={"input_preview": {"tool_input": tool_input}},
        )
        raise


def _tool_config(state: RouterRagState, config: RunnableConfig, *, caller_node: str, tool_name: str) -> RunnableConfig:
    caller_node_id = _runtime_node_id(config, caller_node)
    caller_node_type = _runtime_node_type(config, caller_node)
    caller_capabilities = _runtime_node_capabilities(config) or node_type_capabilities(caller_node_type)
    validate_tool_call_allowed(
        tool_name,
        caller_node_id,
        caller_node_type=caller_node_type,
        caller_capabilities=caller_capabilities,
    )
    contract_id = get_tool_contract_id(tool_name)
    allowed_tool_ids = state.get("allowed_tool_ids")
    if not isinstance(allowed_tool_ids, list) or contract_id not in allowed_tool_ids:
        raise ValueError(
            f"Tool {tool_name} with contract ID {contract_id} is not enabled for this agent run"
        )
    updated = dict(config or {})
    configurable = dict(updated.get("configurable") or {})
    configurable.update(
        {
            "agent_run_id": state.get("agent_run_id"),
            "caller_node": caller_node_id,
            "caller_node_type": caller_node_type,
            "caller_capabilities": caller_capabilities,
            "route": state.get("route"),
            "tool_name": tool_name,
        }
    )
    updated["configurable"] = configurable
    metadata = dict(updated.get("metadata") or {})
    metadata.update(
        {
            "agent_run_id": state.get("agent_run_id"),
            "caller_node": caller_node_id,
            "caller_node_type": caller_node_type,
            "caller_capabilities": caller_capabilities,
            "route": state.get("route"),
            "tool_name": tool_name,
        }
    )
    updated["metadata"] = metadata
    return updated


def _tool_config_for_node(
    state: RouterRagState,
    config: RunnableConfig,
    *,
    caller_node: str,
    tool_name: str,
    started: float,
) -> RunnableConfig:
    try:
        return _tool_config(state, config, caller_node=caller_node, tool_name=tool_name)
    except Exception as exc:
        _append_failed_node_event(
            state,
            config,
            caller_node,
            started,
            exc,
            data={"input_preview": {"tool_name": tool_name}},
        )
        raise


def _log_node_end(
    state: RouterRagState,
    node: str,
    started: float,
    data: Optional[Dict[str, Any]] = None,
) -> None:
    payload = data or {}
    logger.info(
        "Router RAG node completed | run_id=%s thread_id=%s node=%s elapsed_ms=%.1f route=%s evidence_chars=%s document_sources=%s web_sources=%s used_chat_ids=%s",
        state.get("agent_run_id"),
        state.get("thread_id"),
        node,
        (time.perf_counter() - started) * 1000,
        payload.get("route", state.get("route")),
        payload.get("evidence_chars", len(str(state.get("evidence") or ""))),
        payload.get("document_source_count", len(state.get("document_sources") or [])),
        payload.get("web_source_count", len(state.get("web_sources") or [])),
        payload.get("used_chat_id_count", len(state.get("used_chat_ids") or [])),
    )


def _safe_json_object(raw: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw[start:end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
    return {}


def _first_int(*values: Any) -> Optional[int]:
    for value in values:
        if value in (None, ""):
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _llm_result_metadata(
    response: Any,
    *,
    model_name: Optional[str] = None,
    normalized_response: Optional[Dict[str, Any]] = None,
    retry_attempts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    usage = getattr(response, "usage_metadata", None)
    usage = usage if isinstance(usage, dict) else {}
    response_metadata = getattr(response, "response_metadata", None)
    response_metadata = response_metadata if isinstance(response_metadata, dict) else {}
    token_usage = response_metadata.get("token_usage") if isinstance(response_metadata.get("token_usage"), dict) else {}
    output_tokens_details = token_usage.get("completion_tokens_details") if isinstance(token_usage.get("completion_tokens_details"), dict) else {}
    input_tokens_details = token_usage.get("prompt_tokens_details") if isinstance(token_usage.get("prompt_tokens_details"), dict) else {}
    content = getattr(response, "content", "")
    content_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=True) if content else ""
    normalized = normalized_response if isinstance(normalized_response, dict) else {}
    reasoning = str(normalized.get("reasoning") or "")

    token_counts = {
        "prompt": _first_int(usage.get("input_tokens"), token_usage.get("prompt_tokens")),
        "completion": _first_int(usage.get("output_tokens"), token_usage.get("completion_tokens")),
        "total": _first_int(usage.get("total_tokens"), token_usage.get("total_tokens")),
        "reasoning": _first_int(
            usage.get("reasoning_tokens"),
            token_usage.get("reasoning_tokens"),
            output_tokens_details.get("reasoning_tokens"),
        ),
        "cached": _first_int(
            usage.get("cached_tokens"),
            token_usage.get("cached_tokens"),
            input_tokens_details.get("cached_tokens"),
        ),
    }
    token_counts = {key: value for key, value in token_counts.items() if value is not None}

    summary = {
        "model_name": model_name or response_metadata.get("model_name") or response_metadata.get("model"),
        "response_chars": len(content_text),
        "token_counts": token_counts,
        "retry_count": len(retry_attempts or []),
        "retry_attempts": retry_attempts or [],
        "reasoning_available": normalized.get("reasoning_available"),
        "reasoning_format": normalized.get("reasoning_format"),
        "reasoning_chars": len(reasoning) if reasoning else None,
        "reasoning_preview": compact_preview(reasoning, limit=1800) if reasoning else None,
    }
    return {key: value for key, value in summary.items() if value not in (None, "", {}, [])}


def _llm_retry_observer() -> tuple[List[Dict[str, Any]], Callable[[Dict[str, Any]], None]]:
    attempts: List[Dict[str, Any]] = []

    def observe(event: Dict[str, Any]) -> None:
        attempts.append(dict(event))

    return attempts, observe


def _should_skip_worker(state: RouterRagState, worker_node: str) -> bool:
    plan = state.get("execution_plan")
    if not isinstance(plan, list):
        return False
    return worker_node not in plan


def _skipped_worker_update(
    state: RouterRagState,
    config: RunnableConfig,
    worker_node: str,
    started: float,
    reason: str,
) -> Dict[str, Any]:
    data = {
        "status": "skipped",
        "skipped": True,
        "skip_reason": reason,
        "input_preview": {"question": compact_preview(state.get("question"))},
        "input_refs": _state_evidence_refs(state),
        "output_refs": _state_evidence_refs(state),
    }
    _log_node_end(state, worker_node, started, data)
    return {"node_events": _append_event(state, worker_node, data, started=started, config=config)}

def _hitl_gates_from_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    return policy.get("gates") if isinstance(policy.get("gates"), dict) else {}


def _normalize_hitl_gate_policy(gate_id: str, gate_policy: Any) -> Dict[str, Any]:
    gate = dict(gate_policy) if isinstance(gate_policy, dict) else {}
    if gate_id == WEB_APPROVAL_GATE_ID:
        gate.setdefault("mode", "approval")
        gate.setdefault("phase", "before")
        gate.setdefault("target", {"node_id": "web_worker", "node_type": "web_worker"})
        gate.setdefault("interrupt_type", "tool_approval")
        gate.setdefault("title", "Approve web search?")
        gate.setdefault(
            "prompt",
            "This answer needs live web research. Approve web search or continue without it.",
        )
        gate.setdefault("allowed_actions", ["approve", "continue_without"])
        gate.setdefault("default_action", "continue_without")
        gate.setdefault("routes", {"approve": "web_worker", "continue_without": "synthesizer"})
    if gate_id == FINAL_REVIEW_GATE_ID:
        gate.setdefault("mode", "review")
        gate.setdefault("phase", "after")
        gate.setdefault("target", {"node_id": "finalizer", "node_type": "finalizer"})
        gate.setdefault("interrupt_type", "final_answer_review")
        gate.setdefault("title", "Review final answer")
        gate.setdefault("prompt", "Approve this answer before it is saved to the thread.")
        gate.setdefault("allowed_actions", ["approve", "edit", "continue_without", "reject"])
        gate.setdefault("default_action", "approve")
        gate.setdefault("routes", {"approve": "END", "edit": "END", "continue_without": "END"})
        gate.setdefault("editable_fields", ["final_answer"])
    gate.setdefault("mode", "approval")
    gate.setdefault("phase", "before")
    if not isinstance(gate.get("routes"), dict):
        gate["routes"] = {}
    if not isinstance(gate.get("allowed_actions"), list):
        gate["allowed_actions"] = ["approve_selected", "continue_without"] if gate.get("mode") == "choice" else ["approve", "continue_without"]
    if not isinstance(gate.get("default_action"), str):
        gate["default_action"] = "approve_selected" if gate.get("mode") == "choice" else "approve"
    return gate


def _normalize_hitl_actions(gate: Dict[str, Any]) -> List[str]:
    allowed = gate.get("allowed_actions")
    if not isinstance(allowed, list) or not all(isinstance(action, str) for action in allowed):
        allowed = ["approve_selected", "continue_without"] if gate.get("mode") == "choice" else ["approve", "continue_without"]
    allowed = [action for action in allowed if action in {"approve", "approve_selected", "continue_without", "reject", "edit"}]
    return allowed or ["approve", "continue_without"]


def _hitl_option_ids(gate: Dict[str, Any]) -> List[str]:
    options = gate.get("options") if isinstance(gate.get("options"), list) else []
    return [
        str(option.get("id"))
        for option in options
        if isinstance(option, dict) and isinstance(option.get("id"), str) and option.get("id")
    ]


def _hitl_selected_option_ids(decision: Dict[str, Any], gate: Dict[str, Any]) -> List[str]:
    valid_ids = _hitl_option_ids(gate)
    selected = decision.get("selected_option_ids")
    if isinstance(selected, str):
        selected = [selected]
    if not isinstance(selected, list):
        selected = []
    normalized = [str(item) for item in selected if str(item) in valid_ids]
    selection_mode = str(gate.get("selection_mode") or "single")
    if selection_mode == "single" and len(normalized) > 1:
        normalized = normalized[:1]
    return normalized


def _hitl_option_targets(gate: Dict[str, Any], selected_option_ids: List[str]) -> List[str]:
    options = gate.get("options") if isinstance(gate.get("options"), list) else []
    selected = set(selected_option_ids)
    targets: List[str] = []
    for option in options:
        if not isinstance(option, dict) or option.get("id") not in selected:
            continue
        target = option.get("target_node_id")
        if isinstance(target, str) and target not in targets:
            targets.append(target)
    return targets


def with_web_approval_hitl_policy(policy: Any) -> Dict[str, Any]:
    """Return a policy with the reusable before-web approval gate enabled."""

    normalized = deepcopy(policy) if isinstance(policy, dict) else {}
    normalized["enabled"] = True
    gates = dict(normalized.get("gates") or {})
    gates[WEB_APPROVAL_GATE_ID] = _normalize_hitl_gate_policy(
        WEB_APPROVAL_GATE_ID,
        gates.get(WEB_APPROVAL_GATE_ID),
    )
    normalized["gates"] = gates
    return normalized


def normalize_hitl_policy_for_thread_settings(policy: Any, thread_settings: Any = None) -> Dict[str, Any]:
    """Normalize legacy thread-level HITL toggles into the reusable policy contract."""

    normalized = deepcopy(policy) if isinstance(policy, dict) else {}
    if isinstance(thread_settings, dict) and bool(thread_settings.get("hitl_web_approval")):
        return with_web_approval_hitl_policy(normalized)
    return normalized


class NodeRegistry:
    """Registry of safe backend node implementations for compiled v2 patterns."""

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
        if route == "clarify" and not isinstance(clarification_options, list):
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
        return await self._answer_from_context(state, config, node_name="direct_answer")

    async def synthesizer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        return await self._answer_from_context(state, config, node_name="synthesizer")

    async def _answer_from_context(self, state: RouterRagState, config: RunnableConfig, *, node_name: str) -> Dict[str, Any]:
        started = time.perf_counter()
        llm = get_llm(state["llm_model"])
        context, context_source = _final_context_from_state(state)
        if state.get("evaluator_report"):
            context = _combine_evidence(
                context,
                json.dumps(state.get("evaluator_report") or {}, ensure_ascii=True, sort_keys=True),
                label="Evaluator report",
                limit=_evidence_text_limit(state),
            )
        messages = build_final_answer_messages(state, context)
        retry_attempts, retry_observer = _llm_retry_observer()
        prompt_details = prompt_summary(
            "Final Answer Prompt",
            messages["system"],
            messages["human"],
        )
        response = await _invoke_llm_for_node(
            llm.ainvoke,
            [
                SystemMessage(content=messages["system"]),
                HumanMessage(content=messages["human"]),
            ],
            state=state,
            config=config,
            node=node_name,
            started=started,
            retry_observer=retry_observer,
            retry_attempts=retry_attempts,
            model_name=state.get("llm_model"),
            failure_data={
                "input_refs": _state_evidence_refs(state) or _prefetch_refs(state.get("pre_fetch_bundle") or {}),
                "input_preview": {
                    "question": compact_preview(state.get("question")),
                    "context_source": context_source,
                    "context": compact_preview(context),
                },
                "prompt_summary": prompt_details,
            },
        )
        normalized = normalize_ai_response(response)
        data = {
            "status": "completed",
            "input_refs": _state_evidence_refs(state) or _prefetch_refs(state.get("pre_fetch_bundle") or {}),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "context_source": context_source,
                "context": compact_preview(context),
            },
            "prompt_summary": prompt_details,
            "llm_result_summary": {
                "answer_chars": len(normalized["answer"] or ""),
                "reasoning_available": bool(normalized["reasoning_available"]),
                "reasoning_format": normalized["reasoning_format"],
                "llm": _llm_result_metadata(
                    response,
                    model_name=state.get("llm_model"),
                    normalized_response=normalized,
                    retry_attempts=retry_attempts,
                ),
            },
            "answer_chars": len(normalized["answer"] or ""),
            "evidence_chars": len(str(context or "")),
            "output_refs": _state_evidence_refs(state) or _prefetch_refs(state.get("pre_fetch_bundle") or {}),
            "output_preview": {"answer": compact_preview(normalized["answer"])},
        }
        _log_node_end(state, node_name, started, data)
        return {
            "final_answer": normalized["answer"],
            "reasoning": normalized["reasoning"],
            "reasoning_available": normalized["reasoning_available"],
            "reasoning_format": normalized["reasoning_format"],
            "node_events": _append_event(state, node_name, data, started=started, config=config),
        }

    async def finalizer(self, state: RouterRagState, config: RunnableConfig) -> Dict[str, Any]:
        started = time.perf_counter()
        if state.get("clarification_options") and not state.get("final_answer"):
            answer = "I need a bit more clarification. Did you mean:\n" + "\n".join(
                f"- {option}" for option in state["clarification_options"]
            )
            data = {
                "status": "completed",
                "answer_chars": len(answer),
                "output_preview": {
                    "answer": compact_preview(answer),
                    "clarification_options": state.get("clarification_options"),
                },
                "llm_result_summary": {
                    "clarification_option_count": len(state.get("clarification_options") or []),
                },
            }
            _log_node_end(state, "finalizer", started, data)
            return {
                "final_answer": answer,
                "reasoning": "",
                "reasoning_available": False,
                "reasoning_format": "none",
                "node_events": _append_event(state, "finalizer", data, started=started, config=config),
            }
        data = {
            "status": "completed",
            "answer_chars": len(state.get("final_answer") or ""),
            "output_refs": _state_evidence_refs(state),
            "output_preview": {"answer": compact_preview(state.get("final_answer"))},
        }
        _log_node_end(state, "finalizer", started, data)
        return {"node_events": _append_event(state, "finalizer", data, started=started, config=config)}

    async def hitl_gate(
        self,
        state: RouterRagState,
        config: RunnableConfig,
        *,
        node_id: str = WEB_APPROVAL_GATE_ID,
    ) -> Dict[str, Any]:
        """Pause at a reusable human-in-the-loop gate declared by hitl_policy."""

        started = time.perf_counter()
        policy = state.get("hitl_policy") if isinstance(state.get("hitl_policy"), dict) else {}
        gates = _hitl_gates_from_policy(policy)
        gate_policy = _normalize_hitl_gate_policy(node_id, gates.get(node_id))
        enabled = bool(policy.get("enabled")) and gate_policy.get("enabled", True) is not False
        if not enabled:
            routes = dict(state.get("hitl_gate_routes") or {})
            routes[node_id] = "approve"
            return _skipped_worker_update(state, config, node_id, started, "hitl_policy_disabled") | {
                "hitl_gate_route": "approve",
                "hitl_gate_routes": routes,
            }

        mode = str(gate_policy.get("mode") or "approval")
        phase = str(gate_policy.get("phase") or "before")
        target = gate_policy.get("target") if isinstance(gate_policy.get("target"), dict) else {}
        target_node_id = target.get("node_id")
        target_node_type = target.get("node_type")
        interrupt_type = str(gate_policy.get("interrupt_type") or gate_policy.get("type") or ("option_review" if mode == "choice" else "human_review"))
        allowed_actions = _normalize_hitl_actions(gate_policy)
        default_action = str(gate_policy.get("default_action") or "continue_without")
        if default_action not in allowed_actions:
            default_action = "continue_without" if "continue_without" in allowed_actions else allowed_actions[0]
        routes_by_action = gate_policy.get("routes") if isinstance(gate_policy.get("routes"), dict) else {}
        visit_index = _runtime_visit_index(config)
        interrupt_count_key = _hitl_interrupt_count_key(node_id, visit_index)

        interrupt_counts = _hitl_interrupt_counts(state)
        interrupt_limit = _hitl_interrupt_limit(policy, gate_policy)
        if interrupt_limit is not None and _hitl_visit_interrupt_count(interrupt_counts, gate_id=node_id, visit_index=visit_index) >= interrupt_limit:
            route = "continue_without" if "continue_without" in allowed_actions or "continue_without" in routes_by_action else default_action
            gate_routes = dict(state.get("hitl_gate_routes") or {})
            gate_routes[node_id] = route
            update: Dict[str, Any] = {
                "hitl_gate_route": route,
                "hitl_gate_routes": gate_routes,
                "hitl_interrupt_counts": interrupt_counts,
            }
            if route == "continue_without":
                update["evidence"] = _combine_evidence(
                    state.get("evidence"),
                    (
                        "The configured human review interrupt limit was reached. "
                        "Continue without additional gated actions unless already approved by available context."
                    ),
                    label="HITL decision",
                    limit=_evidence_text_limit(state),
                )
            return _skipped_worker_update(state, config, node_id, started, "hitl_interrupt_limit_exhausted") | update

        options = gate_policy.get("options") if isinstance(gate_policy.get("options"), list) else []
        option_ids = _hitl_option_ids(gate_policy)
        input_summary = {
            "question": compact_preview(state.get("question")),
            "route": state.get("route"),
            "route_reason": compact_preview(state.get("route_reason")),
            "document_source_count": len(state.get("document_sources") or []),
            "web_source_count": len(state.get("web_sources") or []),
            "used_chat_id_count": len(state.get("used_chat_ids") or []),
            "evidence": compact_preview(state.get("evidence")),
        }
        proposed_tool = None
        if target_node_id == "web_worker" or node_id == WEB_APPROVAL_GATE_ID:
            proposed_tool = {
                "name": "search_web",
                "caller_node": "web_worker",
                "input": compact_preview(state.get("question"), limit=1000),
            }

        decision = interrupt(
            {
                "gate_id": node_id,
                "node_id": node_id,
                "target_node_id": target_node_id,
                "target_node_type": target_node_type,
                "visit_index": visit_index,
                "interrupt_count_key": interrupt_count_key,
                "phase": phase,
                "mode": mode,
                "type": interrupt_type,
                "title": gate_policy.get("title") or ("Choose approved options" if mode == "choice" else "Human review requested"),
                "prompt": gate_policy.get("prompt")
                or gate_policy.get("body")
                or ("Select which options may run." if mode == "choice" else "Review this step before the graph continues."),
                "allowed_actions": allowed_actions,
                "default_action": default_action,
                "selection_mode": gate_policy.get("selection_mode") if mode == "choice" else None,
                "options": options if mode == "choice" else None,
                "checkpoint_resume": True,
                "reject_behavior": "resume" if "reject" in dict(gate_policy.get("routes") or {}) else gate_policy.get("reject_behavior"),
                "input_summary": input_summary,
                "proposed_tool": proposed_tool,
                "proposed_final_answer": compact_preview(state.get("final_answer"), limit=2000) if mode == "review" else None,
                "editable_fields": gate_policy.get("editable_fields") if mode == "review" else None,
            }
        )
        decision = decision if isinstance(decision, dict) else {"action": str(decision or default_action)}
        action = str(decision.get("action") or default_action)
        if action not in allowed_actions:
            action = default_action
        selected_option_ids = _hitl_selected_option_ids(decision, gate_policy) if mode == "choice" else []
        if action == "approve_selected" and not selected_option_ids and option_ids:
            selected_option_ids = [option_ids[0]]

        if mode == "choice" and action == "approve_selected":
            route: Any = selected_option_ids[0] if selected_option_ids else "continue_without"
        elif action == "approve":
            route = "approve"
        elif action in routes_by_action:
            route = action
        else:
            route = "continue_without" if action in {"continue_without", "reject"} else action

        gate_routes = dict(state.get("hitl_gate_routes") or {})
        gate_routes[node_id] = route
        selected_options_by_gate = dict(state.get("hitl_selected_options") or {})
        if selected_option_ids:
            selected_options_by_gate[node_id] = selected_option_ids

        selected_targets = _hitl_option_targets(gate_policy, selected_option_ids)
        execution_plan = state.get("execution_plan")
        execution_plan_update = None
        if selected_targets and all(target in WORKER_NODE_ORDER for target in selected_targets):
            execution_plan_update = [target for target in WORKER_NODE_ORDER if target in selected_targets]

        update: Dict[str, Any] = {
            "hitl_gate_route": route,
            "hitl_gate_routes": gate_routes,
            "hitl_selected_options": selected_options_by_gate,
            "hitl_interrupt_counts": {
                **interrupt_counts,
                node_id: interrupt_counts.get(node_id, 0) + 1,
                interrupt_count_key: interrupt_counts.get(interrupt_count_key, 0) + 1,
            },
            "hitl_decisions": [
                *(state.get("hitl_decisions") if isinstance(state.get("hitl_decisions"), list) else []),
                {
                    "gate_id": node_id,
                    "node_id": node_id,
                    "target_node_id": target_node_id,
                    "visit_index": visit_index,
                    "interrupt_count_key": interrupt_count_key,
                    "phase": phase,
                    "mode": mode,
                    "type": interrupt_type,
                    "action": action,
                    "selected_option_ids": selected_option_ids,
                    "decision": {
                        key: value
                        for key, value in decision.items()
                        if key not in {"resume_token"}
                    },
                },
            ],
        }
        if execution_plan_update is not None:
            update["execution_plan"] = execution_plan_update
        elif isinstance(execution_plan, list):
            update["execution_plan"] = execution_plan

        if mode == "review":
            update["human_review_decision"] = {
                key: value
                for key, value in decision.items()
                if key not in {"resume_token"}
            }
            edited_payload = decision.get("edited_payload") if isinstance(decision.get("edited_payload"), dict) else {}
            edited_answer = edited_payload.get("final_answer") or edited_payload.get("answer")
            if action == "edit" and isinstance(edited_answer, str) and edited_answer.strip():
                update["final_answer"] = edited_answer.strip()

        if route == "continue_without" or action == "reject":
            update["evidence"] = _combine_evidence(
                state.get("evidence"),
                (
                    "A human reviewer chose to continue without one or more gated options. "
                    "Do not claim skipped tools, branches, or live evidence were checked; answer only from available context."
                ),
                label="HITL decision",
                limit=_evidence_text_limit(state),
            )

        data = {
            "status": "completed",
            "action": action,
            "route": state.get("route"),
            "route_reason": state.get("route_reason"),
            "input_preview": {
                "question": compact_preview(state.get("question")),
                "route": state.get("route"),
                "route_reason": compact_preview(state.get("route_reason")),
                "options": options if mode == "choice" else None,
                "proposed_final_answer": compact_preview(state.get("final_answer")) if mode == "review" else None,
            },
            "output_preview": {
                "decision": update["hitl_decisions"][-1],
                "next": routes_by_action.get(route) or (selected_targets[0] if selected_targets else route),
                "final_answer": compact_preview(update.get("final_answer") or state.get("final_answer")) if mode == "review" else None,
            },
        }
        _log_node_end(state, node_id, started, data)
        return {
            **update,
            "node_events": _append_event(state, node_id, data, started=started, config=config),
        }

class TemplateCompiler(TemplateMaterializer):
    """Compile validated v2 template specs into LangGraph StateGraph instances."""

    def __init__(self, registry: Optional[NodeRegistry] = None):
        self.registry = registry or NodeRegistry()

    def compile(
        self,
        spec: Dict[str, Any],
        *,
        checkpointer: Any = None,
    ):
        from app.agent_workflows.validator import TemplateValidator

        graph_spec = ((spec.get("config") or {}).get("graph") or {}) if isinstance(spec, dict) else {}
        if not graph_spec.get("hitl_compiled"):
            TemplateValidator().validate(spec)
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
