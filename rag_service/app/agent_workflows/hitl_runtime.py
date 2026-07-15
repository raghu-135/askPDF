from __future__ import annotations

import time
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt as _langgraph_interrupt

from app.agent_workflows.evidence import (
    combine_evidence,
    evidence_text_limit,
)
from app.agent_workflows.planning import WORKER_NODE_ORDER
from app.agent_workflows.runtime_invocation import (
    append_event,
    log_node_end,
    skipped_worker_update,
)
from app.agent_workflows.state import RouterRagState, runtime_visit_index
from app.agent_workflows.trace import compact_preview


FINAL_REVIEW_GATE_ID = "human_review_gate"
WEB_APPROVAL_GATE_ID = "web_approval_gate"


def _interrupt(payload: Dict[str, Any]) -> Any:
    graph_module = sys.modules.get("app.agent_workflows.graph")
    interrupt_fn = getattr(graph_module, "interrupt", _langgraph_interrupt)
    return interrupt_fn(payload)


def hitl_interrupt_counts(state: RouterRagState) -> Dict[str, int]:
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


def hitl_interrupt_limit(policy: Dict[str, Any], gate_policy: Dict[str, Any]) -> Optional[int]:
    raw = gate_policy.get("max_interrupts_per_run", policy.get("max_interrupts_per_run"))
    if raw in (None, ""):
        return None
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 1


def hitl_interrupt_count_key(gate_id: str, visit_index: Optional[int]) -> str:
    if isinstance(visit_index, int) and visit_index >= 1:
        return f"{gate_id}:visit:{visit_index}"
    return gate_id


def hitl_visit_interrupt_count(counts: Dict[str, int], *, gate_id: str, visit_index: Optional[int]) -> int:
    visit_key = hitl_interrupt_count_key(gate_id, visit_index)
    has_visit_counts = any(key.startswith(f"{gate_id}:visit:") for key in counts)
    if visit_key != gate_id and has_visit_counts:
        return counts.get(visit_key, 0)
    return counts.get(gate_id, 0)


def hitl_gates_from_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    return policy.get("gates") if isinstance(policy.get("gates"), dict) else {}


def normalize_hitl_gate_policy(gate_id: str, gate_policy: Any) -> Dict[str, Any]:
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


def normalize_hitl_actions(gate: Dict[str, Any]) -> List[str]:
    allowed = gate.get("allowed_actions")
    if not isinstance(allowed, list) or not all(isinstance(action, str) for action in allowed):
        allowed = ["approve_selected", "continue_without"] if gate.get("mode") == "choice" else ["approve", "continue_without"]
    allowed = [action for action in allowed if action in {"approve", "approve_selected", "continue_without", "reject", "edit"}]
    return allowed or ["approve", "continue_without"]


def hitl_option_ids(gate: Dict[str, Any]) -> List[str]:
    options = gate.get("options") if isinstance(gate.get("options"), list) else []
    return [
        str(option.get("id"))
        for option in options
        if isinstance(option, dict) and isinstance(option.get("id"), str) and option.get("id")
    ]


def hitl_selected_option_ids(decision: Dict[str, Any], gate: Dict[str, Any]) -> List[str]:
    valid_ids = hitl_option_ids(gate)
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


def hitl_option_targets(gate: Dict[str, Any], selected_option_ids: List[str]) -> List[str]:
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
    gates[WEB_APPROVAL_GATE_ID] = normalize_hitl_gate_policy(
        WEB_APPROVAL_GATE_ID,
        gates.get(WEB_APPROVAL_GATE_ID),
    )
    normalized["gates"] = gates
    return normalized


def normalize_hitl_policy_for_thread_settings(policy: Any, thread_settings: Any = None) -> Dict[str, Any]:
    """Normalize thread-level HITL toggles into the reusable policy contract."""

    normalized = deepcopy(policy) if isinstance(policy, dict) else {}
    if isinstance(thread_settings, dict) and bool(thread_settings.get("hitl_web_approval")):
        return with_web_approval_hitl_policy(normalized)
    return normalized


async def hitl_gate_node(
    state: RouterRagState,
    config: RunnableConfig,
    *,
    node_id: str = WEB_APPROVAL_GATE_ID,
) -> Dict[str, Any]:
    """Pause at a reusable human-in-the-loop gate declared by hitl_policy."""

    started = time.perf_counter()
    policy = state.get("hitl_policy") if isinstance(state.get("hitl_policy"), dict) else {}
    gates = hitl_gates_from_policy(policy)
    gate_policy = normalize_hitl_gate_policy(node_id, gates.get(node_id))
    enabled = bool(policy.get("enabled")) and gate_policy.get("enabled", True) is not False
    if not enabled:
        routes = dict(state.get("hitl_gate_routes") or {})
        routes[node_id] = "approve"
        return skipped_worker_update(state, config, node_id, started, "hitl_policy_disabled") | {
            "hitl_gate_route": "approve",
            "hitl_gate_routes": routes,
        }

    mode = str(gate_policy.get("mode") or "approval")
    phase = str(gate_policy.get("phase") or "before")
    target = gate_policy.get("target") if isinstance(gate_policy.get("target"), dict) else {}
    target_node_id = target.get("node_id")
    target_node_type = target.get("node_type")
    interrupt_type = str(gate_policy.get("interrupt_type") or gate_policy.get("type") or ("option_review" if mode == "choice" else "human_review"))
    allowed_actions = normalize_hitl_actions(gate_policy)
    default_action = str(gate_policy.get("default_action") or "continue_without")
    if default_action not in allowed_actions:
        default_action = "continue_without" if "continue_without" in allowed_actions else allowed_actions[0]
    routes_by_action = gate_policy.get("routes") if isinstance(gate_policy.get("routes"), dict) else {}
    visit_index = runtime_visit_index(config)
    interrupt_count_key = hitl_interrupt_count_key(node_id, visit_index)

    interrupt_counts = hitl_interrupt_counts(state)
    interrupt_limit = hitl_interrupt_limit(policy, gate_policy)
    if interrupt_limit is not None and hitl_visit_interrupt_count(interrupt_counts, gate_id=node_id, visit_index=visit_index) >= interrupt_limit:
        route = "continue_without" if "continue_without" in allowed_actions or "continue_without" in routes_by_action else default_action
        gate_routes = dict(state.get("hitl_gate_routes") or {})
        gate_routes[node_id] = route
        update: Dict[str, Any] = {
            "hitl_gate_route": route,
            "hitl_gate_routes": gate_routes,
            "hitl_interrupt_counts": interrupt_counts,
        }
        if route == "continue_without":
            update["evidence"] = combine_evidence(
                state.get("evidence"),
                (
                    "The configured human review interrupt limit was reached. "
                    "Continue without additional gated actions unless already approved by available context."
                ),
                label="HITL decision",
                limit=evidence_text_limit(state),
            )
        return skipped_worker_update(state, config, node_id, started, "hitl_interrupt_limit_exhausted") | update

    options = gate_policy.get("options") if isinstance(gate_policy.get("options"), list) else []
    option_ids = hitl_option_ids(gate_policy)
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

    decision = _interrupt(
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
    selected_option_ids = hitl_selected_option_ids(decision, gate_policy) if mode == "choice" else []
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

    selected_targets = hitl_option_targets(gate_policy, selected_option_ids)
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
        update["evidence"] = combine_evidence(
            state.get("evidence"),
            (
                "A human reviewer chose to continue without one or more gated options. "
                "Do not claim skipped tools, branches, or live evidence were checked; answer only from available context."
            ),
            label="HITL decision",
            limit=evidence_text_limit(state),
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
    log_node_end(state, node_id, started, data)
    return {
        **update,
        "node_events": append_event(state, node_id, data, started=started, config=config),
    }
