"""Project a delegated child's pending interrupt onto the parent Auto graph."""

from __future__ import annotations

from typing import Any, Mapping


def delegated_child_interrupt_payload(
    *,
    child_run_id: str,
    child_workflow_id: str,
    pending_interrupt: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a parent-graph interrupt that resumes the child after human review."""

    payload = {
        key: value
        for key, value in pending_interrupt.items()
        if value is not None
    }
    payload.setdefault(
        "type",
        str(pending_interrupt.get("type") or pending_interrupt.get("interrupt_type") or "tool_approval"),
    )
    payload.setdefault("response_operation", pending_interrupt.get("response_operation") or "run.resume")
    payload.setdefault("checkpoint_resume", True)
    payload["delegated_child_run_id"] = child_run_id
    payload["delegated_child_workflow_id"] = child_workflow_id
    payload["gate_id"] = str(pending_interrupt.get("gate_id") or "workflow_delegate")
    payload["node_id"] = "workflow_delegate"
    payload["target_node_id"] = "workflow_delegate"
    payload["target_node_type"] = "workflow_delegate"
    payload.setdefault("title", pending_interrupt.get("title") or "Delegated workflow needs your approval")
    payload.setdefault(
        "prompt",
        pending_interrupt.get("prompt") or "Approve or decline so the delegated agent can continue.",
    )
    if pending_interrupt.get("proposed_tool") is not None:
        payload["proposed_tool"] = pending_interrupt.get("proposed_tool")
    return payload


def delegated_child_resume_request(decision: Mapping[str, Any], child_interrupt: Mapping[str, Any]) -> dict[str, Any]:
    """Map a parent resume decision onto a child run resume request."""

    action = str(
        decision.get("action")
        or decision.get("requested_action")
        or child_interrupt.get("default_action")
        or "approve"
    )
    return {
        "interrupt_id": str(child_interrupt.get("interrupt_id") or ""),
        "action": action,
        "approval_scope": decision.get("approval_scope"),
        "approval_feedback": decision.get("approval_feedback"),
        "approval_modifications": decision.get("approval_modifications"),
        "edited_payload": decision.get("edited_payload"),
        "selected_option_ids": decision.get("selected_option_ids"),
        "client_metadata": decision.get("client_metadata"),
    }