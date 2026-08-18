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
from app.agent_workflows.enums import (
    AgentRunResumeAction,
    GraphSentinel,
    HitlInterruptType,
    HitlMode,
    HitlPhase,
    HitlRejectBehavior,
    HitlSelectionMode,
    HITL_ACTIONS,
    NodeEventStatus,
    ToolName,
    WorkflowNodeType,
)
from app.agent_workflows.planning import WORKER_NODE_ORDER, available_worker_node_ids
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
        gate.setdefault("mode", HitlMode.APPROVAL.value)
        gate.setdefault("phase", HitlPhase.BEFORE.value)
        gate.setdefault(
            "target",
            {
                "node_id": WorkflowNodeType.WEB_WORKER.value,
                "node_type": WorkflowNodeType.WEB_WORKER.value,
            },
        )
        gate.setdefault("interrupt_type", "tool_approval")
        gate.setdefault("title", "Approve web search?")
        gate.setdefault(
            "prompt",
            "This answer needs live web research. Approve web search or continue without it.",
        )
        gate.setdefault("allowed_actions", [
            AgentRunResumeAction.APPROVE.value,
            AgentRunResumeAction.APPROVE_FOR_SCOPE.value,
            AgentRunResumeAction.CONTINUE_WITHOUT.value,
        ])
        gate.setdefault("default_action", AgentRunResumeAction.CONTINUE_WITHOUT.value)
        gate.setdefault(
            "routes",
            {
                AgentRunResumeAction.APPROVE.value: WorkflowNodeType.WEB_WORKER.value,
                AgentRunResumeAction.CONTINUE_WITHOUT.value: WorkflowNodeType.SYNTHESIZER.value,
            },
        )
    if gate_id == FINAL_REVIEW_GATE_ID:
        gate.setdefault("mode", HitlMode.REVIEW.value)
        gate.setdefault("phase", HitlPhase.AFTER.value)
        gate.setdefault(
            "target",
            {
                "node_id": WorkflowNodeType.FINALIZER.value,
                "node_type": WorkflowNodeType.FINALIZER.value,
            },
        )
        gate.setdefault("interrupt_type", "final_answer_review")
        gate.setdefault("title", "Review final answer")
        gate.setdefault("prompt", "Approve this answer before it is saved to the thread.")
        gate.setdefault("allowed_actions", [
            AgentRunResumeAction.APPROVE.value,
            AgentRunResumeAction.EDIT.value,
            AgentRunResumeAction.CONTINUE_WITHOUT.value,
            AgentRunResumeAction.REJECT.value,
        ])
        gate.setdefault("default_action", AgentRunResumeAction.APPROVE.value)
        gate.setdefault("routes", {
            AgentRunResumeAction.APPROVE.value: GraphSentinel.END.value,
            AgentRunResumeAction.EDIT.value: GraphSentinel.END.value,
            AgentRunResumeAction.CONTINUE_WITHOUT.value: GraphSentinel.END.value,
        })
        gate.setdefault("editable_fields", ["final_answer"])
    gate.setdefault("mode", HitlMode.APPROVAL.value)
    gate.setdefault("phase", HitlPhase.BEFORE.value)
    if not isinstance(gate.get("routes"), dict):
        gate["routes"] = {}
    if not isinstance(gate.get("allowed_actions"), list):
        gate["allowed_actions"] = [AgentRunResumeAction.APPROVE_SELECTED.value, AgentRunResumeAction.CONTINUE_WITHOUT.value] if gate.get("mode") == HitlMode.CHOICE.value else [AgentRunResumeAction.APPROVE.value, AgentRunResumeAction.CONTINUE_WITHOUT.value]
    if not isinstance(gate.get("default_action"), str):
        gate["default_action"] = AgentRunResumeAction.APPROVE_SELECTED.value if gate.get("mode") == HitlMode.CHOICE.value else AgentRunResumeAction.APPROVE.value
    return gate


def normalize_hitl_actions(gate: Dict[str, Any]) -> List[str]:
    allowed = gate.get("allowed_actions")
    if not isinstance(allowed, list) or not all(isinstance(action, str) for action in allowed):
        allowed = [AgentRunResumeAction.APPROVE_SELECTED.value, AgentRunResumeAction.CONTINUE_WITHOUT.value] if gate.get("mode") == HitlMode.CHOICE.value else [AgentRunResumeAction.APPROVE.value, AgentRunResumeAction.CONTINUE_WITHOUT.value]
    allowed = [action for action in allowed if action in HITL_ACTIONS]
    return allowed or [AgentRunResumeAction.APPROVE.value, AgentRunResumeAction.CONTINUE_WITHOUT.value]


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
    selection_mode = str(gate.get("selection_mode") or HitlSelectionMode.SINGLE.value)
    if selection_mode == HitlSelectionMode.SINGLE.value and len(normalized) > 1:
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
        routes[node_id] = AgentRunResumeAction.APPROVE.value
        return skipped_worker_update(state, config, node_id, started, "hitl_policy_disabled") | {
            "hitl_gate_route": AgentRunResumeAction.APPROVE.value,
            "hitl_gate_routes": routes,
        }

    if node_id == WEB_APPROVAL_GATE_ID:
        web_worker_ids = {
            str(item.get("id")) for item in state.get("available_worker_nodes") or []
            if isinstance(item, dict) and item.get("type") == WorkflowNodeType.WEB_WORKER.value
        }
        proposals = [item for item in state.get("work_item_proposals") or [] if isinstance(item, dict)]
        web_planned = any(
            item.get("worker_node_id") in web_worker_ids
            or item.get("worker_type") == WorkflowNodeType.WEB_WORKER.value
            for item in proposals
        )
        if proposals and not web_planned:
            routes = dict(state.get("hitl_gate_routes") or {})
            routes[node_id] = AgentRunResumeAction.APPROVE.value
            return skipped_worker_update(state, config, node_id, started, "web_not_planned") | {
                "hitl_gate_route": AgentRunResumeAction.APPROVE.value,
                "hitl_gate_routes": routes,
            }
        grant = (state.get("hitl_approval_grants") or {}).get(node_id)
        grant_status = str(grant.get("status") or "") if isinstance(grant, dict) else ""
        if grant_status in {"allowed", "denied"}:
            route = (
                AgentRunResumeAction.APPROVE.value
                if grant_status == "allowed"
                else AgentRunResumeAction.CONTINUE_WITHOUT.value
            )
            routes = dict(state.get("hitl_gate_routes") or {})
            routes[node_id] = route
            update = {
                "hitl_gate_route": route,
                "hitl_gate_routes": routes,
            }
            if grant_status == "denied":
                update["work_item_proposals"] = [
                    item for item in proposals
                    if item.get("worker_node_id") not in web_worker_ids
                ]
                update["execution_plan"] = [
                    item for item in state.get("execution_plan") or []
                    if item not in web_worker_ids
                ]
            return skipped_worker_update(
                state,
                config,
                node_id,
                started,
                f"web_{grant_status}_for_run",
            ) | update

    mode = str(gate_policy.get("mode") or HitlMode.APPROVAL.value)
    phase = str(gate_policy.get("phase") or HitlPhase.BEFORE.value)
    target = gate_policy.get("target") if isinstance(gate_policy.get("target"), dict) else {}
    target_node_id = target.get("node_id")
    target_node_type = target.get("node_type")
    default_interrupt_type = (
        HitlInterruptType.OPTION_REVIEW.value
        if mode == HitlMode.CHOICE.value
        else HitlInterruptType.HUMAN_REVIEW.value
    )
    interrupt_type = str(gate_policy.get("interrupt_type") or gate_policy.get("type") or default_interrupt_type)
    allowed_actions = normalize_hitl_actions(gate_policy)
    default_action = str(gate_policy.get("default_action") or AgentRunResumeAction.CONTINUE_WITHOUT.value)
    if default_action not in allowed_actions:
        default_action = AgentRunResumeAction.CONTINUE_WITHOUT.value if AgentRunResumeAction.CONTINUE_WITHOUT.value in allowed_actions else allowed_actions[0]
    routes_by_action = gate_policy.get("routes") if isinstance(gate_policy.get("routes"), dict) else {}
    visit_index = runtime_visit_index(config)
    interrupt_count_key = hitl_interrupt_count_key(node_id, visit_index)

    interrupt_counts = hitl_interrupt_counts(state)
    interrupt_limit = hitl_interrupt_limit(policy, gate_policy)
    if interrupt_limit is not None and hitl_visit_interrupt_count(interrupt_counts, gate_id=node_id, visit_index=visit_index) >= interrupt_limit:
        route = AgentRunResumeAction.CONTINUE_WITHOUT.value if AgentRunResumeAction.CONTINUE_WITHOUT.value in allowed_actions or AgentRunResumeAction.CONTINUE_WITHOUT.value in routes_by_action else default_action
        gate_routes = dict(state.get("hitl_gate_routes") or {})
        gate_routes[node_id] = route
        update: Dict[str, Any] = {
            "hitl_gate_route": route,
            "hitl_gate_routes": gate_routes,
            "hitl_interrupt_counts": interrupt_counts,
        }
        if route == AgentRunResumeAction.CONTINUE_WITHOUT.value:
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
    if target_node_id == WorkflowNodeType.WEB_WORKER.value or node_id == WEB_APPROVAL_GATE_ID:
        proposed_tool = {
            "name": ToolName.SEARCH_WEB.value,
            "caller_node": WorkflowNodeType.WEB_WORKER.value,
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
            "title": gate_policy.get("title") or ("Choose approved options" if mode == HitlMode.CHOICE.value else "Human review requested"),
            "prompt": gate_policy.get("prompt")
            or gate_policy.get("body")
            or ("Select which options may run." if mode == HitlMode.CHOICE.value else "Review this step before the graph continues."),
            "allowed_actions": allowed_actions,
            "default_action": default_action,
            "selection_mode": gate_policy.get("selection_mode") if mode == HitlMode.CHOICE.value else None,
            "options": options if mode == HitlMode.CHOICE.value else None,
            "checkpoint_resume": True,
            "reject_behavior": HitlRejectBehavior.RESUME.value if AgentRunResumeAction.REJECT.value in dict(gate_policy.get("routes") or {}) else gate_policy.get("reject_behavior"),
            "input_summary": input_summary,
            "proposed_tool": proposed_tool,
            "proposed_final_answer": compact_preview(state.get("final_answer"), limit=2000) if mode == HitlMode.REVIEW.value else None,
            "editable_fields": gate_policy.get("editable_fields") if mode == HitlMode.REVIEW.value else None,
        }
    )
    decision = decision if isinstance(decision, dict) else {"action": str(decision or default_action)}
    action = str(decision.get("action") or default_action)
    if action not in allowed_actions:
        action = default_action
    selected_option_ids = hitl_selected_option_ids(decision, gate_policy) if mode == HitlMode.CHOICE.value else []
    if action == AgentRunResumeAction.APPROVE_SELECTED.value and not selected_option_ids and option_ids:
        selected_option_ids = [option_ids[0]]

    if mode == HitlMode.CHOICE.value and action == AgentRunResumeAction.APPROVE_SELECTED.value:
        route: Any = selected_option_ids[0] if selected_option_ids else AgentRunResumeAction.CONTINUE_WITHOUT.value
    elif action in {AgentRunResumeAction.APPROVE.value, AgentRunResumeAction.APPROVE_FOR_SCOPE.value}:
        route = AgentRunResumeAction.APPROVE.value
    elif action in routes_by_action:
        route = action
    else:
        route = AgentRunResumeAction.CONTINUE_WITHOUT.value if action in {AgentRunResumeAction.CONTINUE_WITHOUT.value, AgentRunResumeAction.REJECT.value} else action

    gate_routes = dict(state.get("hitl_gate_routes") or {})
    gate_routes[node_id] = route
    selected_options_by_gate = dict(state.get("hitl_selected_options") or {})
    if selected_option_ids:
        selected_options_by_gate[node_id] = selected_option_ids

    selected_targets = hitl_option_targets(gate_policy, selected_option_ids)
    execution_plan = state.get("execution_plan")
    execution_plan_update = None
    worker_order = available_worker_node_ids(state.get("available_worker_nodes")) or WORKER_NODE_ORDER
    if selected_targets and all(target in worker_order for target in selected_targets):
        execution_plan_update = [target for target in worker_order if target in selected_targets]

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
    approval_grants = dict(state.get("hitl_approval_grants") or {})
    if node_id == WEB_APPROVAL_GATE_ID and action == AgentRunResumeAction.APPROVE_FOR_SCOPE.value:
        approval_grants[node_id] = {"status": "allowed", "scope": "run"}
        update["hitl_approval_grants"] = approval_grants
    elif node_id == WEB_APPROVAL_GATE_ID and action in {
        AgentRunResumeAction.CONTINUE_WITHOUT.value,
        AgentRunResumeAction.REJECT.value,
    }:
        approval_grants[node_id] = {"status": "denied", "scope": "run"}
        update["hitl_approval_grants"] = approval_grants
    if execution_plan_update is not None:
        update["execution_plan"] = execution_plan_update
    elif isinstance(execution_plan, list):
        update["execution_plan"] = execution_plan

    if node_id == WEB_APPROVAL_GATE_ID and action == AgentRunResumeAction.CONTINUE_WITHOUT.value:
        web_worker_ids = {
            str(item.get("id")) for item in state.get("available_worker_nodes") or []
            if isinstance(item, dict) and item.get("type") == WorkflowNodeType.WEB_WORKER.value
        }
        update["work_item_proposals"] = [
            item for item in state.get("work_item_proposals") or []
            if isinstance(item, dict) and item.get("worker_node_id") not in web_worker_ids
        ]
        update["execution_plan"] = [
            item for item in state.get("execution_plan") or [] if item not in web_worker_ids
        ]

    if mode == HitlMode.REVIEW.value:
        update["human_review_decision"] = {
            key: value
            for key, value in decision.items()
            if key not in {"resume_token"}
        }
        edited_payload = decision.get("edited_payload") if isinstance(decision.get("edited_payload"), dict) else {}
        edited_answer = edited_payload.get("final_answer") or edited_payload.get("answer")
        if action == AgentRunResumeAction.EDIT.value and isinstance(edited_answer, str) and edited_answer.strip():
            update["final_answer"] = edited_answer.strip()

    if route == AgentRunResumeAction.CONTINUE_WITHOUT.value or action == AgentRunResumeAction.REJECT.value:
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
        "status": NodeEventStatus.COMPLETED.value,
        "action": action,
        "route": state.get("route"),
        "route_reason": state.get("route_reason"),
        "input_preview": {
            "question": compact_preview(state.get("question")),
            "route": state.get("route"),
            "route_reason": compact_preview(state.get("route_reason")),
            "options": options if mode == HitlMode.CHOICE.value else None,
            "proposed_final_answer": compact_preview(state.get("final_answer")) if mode == HitlMode.REVIEW.value else None,
        },
        "output_preview": {
            "decision": update["hitl_decisions"][-1],
            "next": routes_by_action.get(route) or (selected_targets[0] if selected_targets else route),
            "final_answer": compact_preview(update.get("final_answer") or state.get("final_answer")) if mode == HitlMode.REVIEW.value else None,
        },
    }
    log_node_end(state, node_id, started, data)
    return {
        **update,
        "node_events": append_event(state, node_id, data, started=started, config=config),
    }
