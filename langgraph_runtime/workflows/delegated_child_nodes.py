"""Auto delegation helpers: parent interrupt while a child awaits human review."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from langchain_core.runnables import RunnableConfig

from langgraph_runtime.control_plane_client import delegate_child_resume_sync
from langgraph_runtime.workflows.hitl_runtime import _interrupt
from runtime_protocol.delegated_child_interrupt import (
    delegated_child_interrupt_payload,
    delegated_child_resume_request,
)


async def await_delegated_child_human_review(
    *,
    response: Mapping[str, Any],
    child_run_id: str,
    child_workflow_id: str,
    parent_run_id: str,
    thread_id: str,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Pause the parent on the child's pending tool or graph interrupt, then finish the child."""

    child_result = response.get("result") if isinstance(response.get("result"), dict) else {}
    pending = response.get("pending_interrupt") or child_result.get("pending_interrupt") or {}
    if not isinstance(pending, dict) or not pending.get("interrupt_id"):
        return dict(response)
    interrupt_payload = delegated_child_interrupt_payload(
        child_run_id=child_run_id,
        child_workflow_id=child_workflow_id,
        pending_interrupt=pending,
    )
    decision = _interrupt(interrupt_payload)
    decision_dict = decision if isinstance(decision, dict) else {"action": str(decision or "approve")}
    resume_body = delegated_child_resume_request(decision_dict, pending)
    if not resume_body.get("interrupt_id"):
        return dict(response)
    return await delegate_child_resume_sync(
        parent_run_id=parent_run_id,
        thread_id=thread_id,
        child_run_id=child_run_id,
        config=config,
        **resume_body,
    )