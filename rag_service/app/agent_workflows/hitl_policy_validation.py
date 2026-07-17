from __future__ import annotations

from typing import Any

from app.agent_workflows.enums import (
    AgentRunResumeAction,
    GraphSentinel,
    HitlMode,
    HitlPhase,
    HITL_ACTIONS,
    HITL_MODES,
    HITL_PHASES,
    HITL_SELECTION_MODES,
)
HITL_GATE_KEYS = {
    "enabled",
    "title",
    "prompt",
    "body",
    "allowed_actions",
    "default_action",
    "target",
    "phase",
    "mode",
    "interrupt_type",
    "type",
    "selection_mode",
    "options",
    "routes",
    "conditions",
    "payload_projection",
    "editable_fields",
    "requires_reason",
    "reject_behavior",
    "max_interrupts_per_run",
}


def collect_hitl_policy_errors(hitl_policy: Any, workflow_id: Any, graph: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(hitl_policy, dict):
        return ["hitl_policy must be an object"]
    unknown_policy_keys = sorted(set(hitl_policy) - {"enabled", "gates", "max_interrupts_per_run"})
    if unknown_policy_keys:
        errors.append(f"hitl_policy only supports keys: enabled, gates, max_interrupts_per_run; unknown: {', '.join(unknown_policy_keys)}")
    if "enabled" in hitl_policy and not isinstance(hitl_policy["enabled"], bool):
        errors.append("hitl_policy.enabled must be a boolean")
    if "max_interrupts_per_run" in hitl_policy:
        try:
            max_interrupts = int(hitl_policy.get("max_interrupts_per_run"))
        except (TypeError, ValueError):
            max_interrupts = 0
        if max_interrupts < 1:
            errors.append("hitl_policy.max_interrupts_per_run must be a positive integer")
    gates = hitl_policy.get("gates", {})
    if not isinstance(gates, dict):
        errors.append("hitl_policy.gates must be an object")
        return errors

    graph_nodes = graph.get("nodes") if isinstance(graph, dict) else []
    node_types = {
        node.get("id"): node.get("type")
        for node in graph_nodes
        if isinstance(node, dict) and isinstance(node.get("id"), str) and isinstance(node.get("type"), str)
    }
    for gate_id, gate in gates.items():
        if not isinstance(gate_id, str) or not gate_id:
            errors.append("hitl_policy.gates keys must be non-empty strings")
            continue
        if not isinstance(gate, dict):
            errors.append(f"hitl_policy.gates.{gate_id} must be an object")
            continue

        unknown_gate_keys = sorted(set(gate) - HITL_GATE_KEYS)
        if unknown_gate_keys:
            errors.append(f"hitl_policy.gates.{gate_id} has unknown keys: {', '.join(unknown_gate_keys)}")
        if "enabled" in gate and not isinstance(gate["enabled"], bool):
            errors.append(f"hitl_policy.gates.{gate_id}.enabled must be a boolean")

        mode = str(gate.get("mode") or HitlMode.APPROVAL.value)
        if mode not in HITL_MODES:
            errors.append(f"hitl_policy.gates.{gate_id}.mode must be one of: {', '.join(sorted(HITL_MODES))}")
        phase = str(gate.get("phase") or HitlPhase.BEFORE.value)
        if phase not in HITL_PHASES:
            errors.append(f"hitl_policy.gates.{gate_id}.phase must be one of: {', '.join(sorted(HITL_PHASES))}")
        if phase == HitlPhase.INSIDE_TOOL.value:
            errors.append(f"hitl_policy.gates.{gate_id}.phase inside_tool is reserved for tool wrappers")

        target = gate.get("target")
        if not isinstance(target, dict):
            errors.append(f"hitl_policy.gates.{gate_id}.target must be an object")
        else:
            target_node_id = target.get("node_id")
            target_node_type = target.get("node_type")
            if target_node_id is not None and not isinstance(target_node_id, str):
                errors.append(f"hitl_policy.gates.{gate_id}.target.node_id must be a string")
            if target_node_type is not None and not isinstance(target_node_type, str):
                errors.append(f"hitl_policy.gates.{gate_id}.target.node_type must be a string")
            if not target_node_id and not target_node_type:
                errors.append(f"hitl_policy.gates.{gate_id}.target must include node_id or node_type")
            if isinstance(target_node_id, str) and target_node_id not in node_types:
                errors.append(f"hitl_policy.gates.{gate_id}.target.node_id is unknown: {target_node_id}")
            if isinstance(target_node_type, str) and target_node_type not in set(node_types.values()):
                errors.append(f"hitl_policy.gates.{gate_id}.target.node_type is unknown: {target_node_type}")

        for key in ("title", "prompt", "body", "default_action", "interrupt_type", "type", "selection_mode", "reject_behavior"):
            if key in gate and not isinstance(gate[key], str):
                errors.append(f"hitl_policy.gates.{gate_id}.{key} must be a string")
        if "max_interrupts_per_run" in gate:
            try:
                max_interrupts = int(gate.get("max_interrupts_per_run"))
            except (TypeError, ValueError):
                max_interrupts = 0
            if max_interrupts < 1:
                errors.append(f"hitl_policy.gates.{gate_id}.max_interrupts_per_run must be a positive integer")
        if "selection_mode" in gate and isinstance(gate.get("selection_mode"), str) and gate["selection_mode"] not in HITL_SELECTION_MODES:
            errors.append(f"hitl_policy.gates.{gate_id}.selection_mode is unsupported")

        allowed_actions = gate.get("allowed_actions", [])
        if not isinstance(allowed_actions, list) or not all(isinstance(action, str) for action in allowed_actions):
            errors.append(f"hitl_policy.gates.{gate_id}.allowed_actions must be a list of strings")
            allowed_actions = []
        else:
            unsupported = sorted(set(allowed_actions) - HITL_ACTIONS)
            if unsupported:
                errors.append(f"hitl_policy.gates.{gate_id}.allowed_actions unsupported: {', '.join(unsupported)}")
            if mode == HitlMode.CHOICE.value and AgentRunResumeAction.APPROVE_SELECTED.value not in allowed_actions:
                errors.append(f"hitl_policy.gates.{gate_id}.allowed_actions must include approve_selected for choice gates")
            if mode == HitlMode.APPROVAL.value and AgentRunResumeAction.APPROVE.value not in allowed_actions:
                errors.append(f"hitl_policy.gates.{gate_id}.allowed_actions must include approve for approval gates")
        default_action = gate.get("default_action")
        if isinstance(default_action, str) and allowed_actions and default_action not in allowed_actions:
            errors.append(f"hitl_policy.gates.{gate_id}.default_action must be in allowed_actions")

        routes = gate.get("routes", {})
        if routes is not None and not isinstance(routes, dict):
            errors.append(f"hitl_policy.gates.{gate_id}.routes must be an object")
        elif isinstance(routes, dict):
            for route_name, route_target in routes.items():
                if not isinstance(route_name, str) or not isinstance(route_target, str):
                    errors.append(f"hitl_policy.gates.{gate_id}.routes keys and values must be strings")
                elif route_target not in node_types and route_target != GraphSentinel.END.value:
                    errors.append(f"hitl_policy.gates.{gate_id}.routes.{route_name} target is unknown: {route_target}")

        options = gate.get("options", [])
        if mode == HitlMode.CHOICE.value:
            if not isinstance(options, list) or not options:
                errors.append(f"hitl_policy.gates.{gate_id}.options must be a non-empty list for choice gates")
            else:
                seen_option_ids: set[str] = set()
                for option in options:
                    if not isinstance(option, dict):
                        errors.append(f"hitl_policy.gates.{gate_id}.options entries must be objects")
                        continue
                    option_id = option.get("id")
                    target_node_id = option.get("target_node_id")
                    if not isinstance(option_id, str) or not option_id:
                        errors.append(f"hitl_policy.gates.{gate_id}.options entries require string id")
                    elif option_id in seen_option_ids:
                        errors.append(f"hitl_policy.gates.{gate_id}.options duplicate id: {option_id}")
                    else:
                        seen_option_ids.add(option_id)
                    if not isinstance(target_node_id, str) or target_node_id not in node_types:
                        errors.append(f"hitl_policy.gates.{gate_id}.options.{option_id or '?'} target_node_id is unknown")
    return errors
