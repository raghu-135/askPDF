from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.agent_workflows.enums import AgentRunResumeAction, HitlMode, HitlPhase, RouteFunctionId


FINAL_REVIEW_GATE_ID = "human_review_gate"
WEB_APPROVAL_GATE_ID = "web_approval_gate"


def _hitl_gates_from_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    return policy.get("gates") if isinstance(policy.get("gates"), dict) else {}


def _normalize_hitl_gate_policy(gate_id: str, gate_policy: Any) -> Dict[str, Any]:
    gate = dict(gate_policy) if isinstance(gate_policy, dict) else {}
    if gate_id == WEB_APPROVAL_GATE_ID:
        gate.setdefault("mode", HitlMode.APPROVAL.value)
        gate.setdefault("phase", HitlPhase.BEFORE.value)
        gate.setdefault("target", {"node_id": "web_worker", "node_type": "web_worker"})
        gate.setdefault("interrupt_type", "tool_approval")
        gate.setdefault("title", "Approve web search?")
        gate.setdefault(
            "prompt",
            "This answer needs live web research. Approve web search or continue without it.",
        )
        gate.setdefault("allowed_actions", [AgentRunResumeAction.APPROVE.value, AgentRunResumeAction.CONTINUE_WITHOUT.value])
        gate.setdefault("default_action", AgentRunResumeAction.CONTINUE_WITHOUT.value)
        gate.setdefault("routes", {AgentRunResumeAction.APPROVE.value: "web_worker", AgentRunResumeAction.CONTINUE_WITHOUT.value: "synthesizer"})
    if gate_id == FINAL_REVIEW_GATE_ID:
        gate.setdefault("mode", HitlMode.REVIEW.value)
        gate.setdefault("phase", HitlPhase.AFTER.value)
        gate.setdefault("target", {"node_id": "finalizer", "node_type": "finalizer"})
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
            AgentRunResumeAction.APPROVE.value: "END",
            AgentRunResumeAction.EDIT.value: "END",
            AgentRunResumeAction.CONTINUE_WITHOUT.value: "END",
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


def resolve_hitl_target_node_id(gate: Dict[str, Any], node_types: Dict[str, str]) -> Optional[str]:
    target = gate.get("target") if isinstance(gate.get("target"), dict) else {}
    node_id = target.get("node_id")
    if isinstance(node_id, str) and node_id in node_types:
        return node_id
    node_type = target.get("node_type")
    if isinstance(node_type, str):
        matches = [candidate for candidate, candidate_type in node_types.items() if candidate_type == node_type]
        if len(matches) == 1:
            return matches[0]
    return None


def default_bypass_target(target_node_id: str, edges: List[Dict[str, Any]]) -> str:
    for edge in edges:
        if edge.get("from") == target_node_id and isinstance(edge.get("to"), str):
            return str(edge["to"])
    return "END"


def hitl_gate_routes(gate: Dict[str, Any], target_node_id: str, edges: List[Dict[str, Any]], *, phase: str) -> Dict[str, str]:
    configured = dict(gate.get("routes") or {})
    mode = str(gate.get("mode") or HitlMode.APPROVAL.value)
    bypass = default_bypass_target(target_node_id, edges)
    routes: Dict[str, str] = {}
    if mode == HitlMode.CHOICE.value:
        options = gate.get("options") if isinstance(gate.get("options"), list) else []
        for option in options:
            if not isinstance(option, dict):
                continue
            option_id = option.get("id")
            option_target = option.get("target_node_id")
            if isinstance(option_id, str) and isinstance(option_target, str):
                routes[option_id] = option_target
        routes[AgentRunResumeAction.CONTINUE_WITHOUT.value] = configured.get(AgentRunResumeAction.CONTINUE_WITHOUT.value) or bypass
        if AgentRunResumeAction.REJECT.value in configured:
            routes[AgentRunResumeAction.REJECT.value] = configured[AgentRunResumeAction.REJECT.value]
        return routes
    routes[AgentRunResumeAction.APPROVE.value] = configured.get(AgentRunResumeAction.APPROVE.value) or (target_node_id if phase == HitlPhase.BEFORE.value else bypass)
    routes[AgentRunResumeAction.CONTINUE_WITHOUT.value] = configured.get(AgentRunResumeAction.CONTINUE_WITHOUT.value) or bypass
    if AgentRunResumeAction.REJECT.value in configured:
        routes[AgentRunResumeAction.REJECT.value] = configured[AgentRunResumeAction.REJECT.value]
    if AgentRunResumeAction.EDIT.value in configured:
        routes[AgentRunResumeAction.EDIT.value] = configured[AgentRunResumeAction.EDIT.value]
    return routes


def insert_before_gate(edges: List[Dict[str, Any]], gate_id: str, target_node_id: str) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    for edge in edges:
        edge = dict(edge)
        if edge.get("conditional") and isinstance(edge.get("routes"), dict):
            routes = dict(edge["routes"])
            changed = False
            for route, target in list(routes.items()):
                if target == target_node_id:
                    routes[route] = gate_id
                    changed = True
            if changed:
                edge["routes"] = routes
            updated.append(edge)
            continue
        if edge.get("to") == target_node_id:
            edge["to"] = gate_id
        updated.append(edge)
    return updated


def insert_after_gate(edges: List[Dict[str, Any]], gate_id: str, target_node_id: str) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    for edge in edges:
        edge = dict(edge)
        if edge.get("from") == target_node_id:
            edge["from"] = gate_id
        updated.append(edge)
    updated.append({"from": target_node_id, "to": gate_id})
    return updated


def materialize_hitl_gates(graph_spec: Dict[str, Any], *, hitl_policy: Dict[str, Any]) -> Dict[str, Any]:
    nodes = [dict(node) for node in graph_spec.get("nodes", []) if isinstance(node, dict)]
    edges = [dict(edge) for edge in graph_spec.get("edges", []) if isinstance(edge, dict)]
    if graph_spec.get("hitl_compiled") or not bool(hitl_policy.get("enabled")):
        return {**graph_spec, "nodes": nodes, "edges": edges, "hitl_compiled": True}

    node_types = {
        str(node.get("id")): str(node.get("type"))
        for node in nodes
        if isinstance(node.get("id"), str) and isinstance(node.get("type"), str)
    }
    existing_node_ids = set(node_types)
    gates = _hitl_gates_from_policy(hitl_policy)
    for gate_id, raw_gate in gates.items():
        if not isinstance(gate_id, str) or gate_id in existing_node_ids:
            continue
        gate = _normalize_hitl_gate_policy(gate_id, raw_gate)
        if gate.get("enabled", True) is False:
            continue
        phase = str(gate.get("phase") or HitlPhase.BEFORE.value)
        if phase == HitlPhase.INSIDE_TOOL.value:
            continue
        target_node_id = resolve_hitl_target_node_id(gate, node_types)
        if not target_node_id:
            continue

        nodes.append({"id": gate_id, "type": "hitl_gate"})
        existing_node_ids.add(gate_id)
        routes = hitl_gate_routes(gate, target_node_id, edges, phase=phase)
        if phase == HitlPhase.BEFORE.value:
            edges = insert_before_gate(edges, gate_id, target_node_id)
        elif phase == HitlPhase.AFTER.value:
            edges = insert_after_gate(edges, gate_id, target_node_id)
        else:
            continue
        edges.append({"from": gate_id, "conditional": True, "route_fn": RouteFunctionId.HITL_GATE.value, "routes": routes})

    return {"nodes": nodes, "edges": edges, "hitl_compiled": True}
