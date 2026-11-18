"""Merge parent tool-approval policy into a child workflow without parent graph gates."""

from __future__ import annotations

from typing import Any, Mapping


def child_workflow_gate_ids(spec: Mapping[str, Any] | None) -> frozenset[str]:
    if not isinstance(spec, Mapping):
        return frozenset()
    config = spec.get("config") if isinstance(spec.get("config"), Mapping) else {}
    policy = config.get("hitl_policy") if isinstance(config.get("hitl_policy"), Mapping) else {}
    gates = policy.get("gates") if isinstance(policy.get("gates"), Mapping) else {}
    return frozenset(str(key) for key in gates if str(key).strip())


def merge_child_delegate_hitl_policy(
    child_policy: Any,
    inherited_policy: Mapping[str, Any] | None,
    *,
    child_gate_ids: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Keep child workflow gates; inherit only tools and interrupt limits from the parent."""

    merged: dict[str, Any] = dict(child_policy) if isinstance(child_policy, Mapping) else {}
    inherited = dict(inherited_policy) if isinstance(inherited_policy, Mapping) else {}
    if inherited.get("tools"):
        tools = dict(merged.get("tools") or {})
        tools.update(dict(inherited["tools"]))
        merged["tools"] = tools
    if "max_interrupts_per_run" in inherited:
        merged["max_interrupts_per_run"] = inherited["max_interrupts_per_run"]
    if child_gate_ids:
        gates = merged.get("gates") if isinstance(merged.get("gates"), Mapping) else {}
        merged["gates"] = {
            gate_id: dict(gate)
            for gate_id, gate in gates.items()
            if gate_id in child_gate_ids and isinstance(gate, Mapping)
        }
        if merged["gates"]:
            merged["enabled"] = bool(merged.get("enabled", True))
    return merged