"""Framework-neutral execution-operation normalization."""

from __future__ import annotations

from typing import Any, Mapping

from app.agent_workflows.trace_sanitization import _bounded_value


OPERATION_KINDS = {
    "operation.started", "operation.completed", "operation.failed", "operation.skipped"
}


def normalize_runtime_event(kind: str, payload: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    data = dict(payload or {})
    if kind in {"interrupt.created", "run.interrupted"}:
        data.setdefault("source_event", kind)
        return "interrupt.requested", dict(_bounded_value(data))
    if kind == "approval.request":
        data.setdefault("source_event", kind)
        return "approval.requested", dict(_bounded_value(data))
    if kind == "approval.response":
        data.setdefault("source_event", kind)
        return "approval.responded", dict(_bounded_value(data))
    if kind.startswith("node."):
        suffix = kind.split(".", 1)[1]
        normalized_kind = f"operation.{suffix}"
        operation_id = str(data.get("operation_id") or data.get("node_id") or data.get("node") or "operation")
        operation_type = str(data.get("operation_type") or data.get("node_type") or operation_id)
        topology_ref = data.get("topology_ref")
        if not isinstance(topology_ref, Mapping):
            topology_ref = {"kind": "graph_node", "id": operation_id}
        # Node events from any runtime are projected into the neutral operation
        # vocabulary. Framework metadata, checkpoint state, and raw graph
        # details are deliberately not reconstructed in the control plane.
        data = {
            **data,
            "operation_id": operation_id,
            "operation_type": operation_type,
            "operation_label": data.get("operation_label") or data.get("label"),
            "visit_index": max(1, int(data.get("visit_index") or 1)),
            "topology_ref": dict(topology_ref),
        }
        for private_key in (
            "checkpoint",
            "checkpoint_before",
            "checkpoint_after",
            "framework_metadata",
            "graph",
            "state",
        ):
            data.pop(private_key, None)
        kind = normalized_kind
    elif kind in OPERATION_KINDS:
        operation_id = str(data.get("operation_id") or "operation")
        data.setdefault("operation_type", "runtime_operation")
        data.setdefault("visit_index", 1)
        data["operation_id"] = operation_id
    return kind, dict(_bounded_value(data))
