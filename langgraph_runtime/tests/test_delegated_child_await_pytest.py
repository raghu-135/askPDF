"""Auto workflow_delegate pauses on delegated child HITL instead of placeholder answers."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from langgraph_runtime.workflows.auto_nodes import workflow_delegate_node


@pytest.mark.asyncio
async def test_workflow_delegate_waits_on_delegated_child_awaiting_human(monkeypatch):
    interrupt_payload: dict = {}

    def fake_interrupt(payload):
        interrupt_payload.update(payload)
        return {"action": "approve", "approval_scope": "once"}

    async def fake_delegate_sync(**_kwargs):
        return {
            "child_run_id": "child-7",
            "status": "awaiting_human",
            "pending_interrupt": {"interrupt_id": "int-7", "type": "tool_approval"},
            "result": {
                "status": "awaiting_human",
                "pending_interrupt": {"interrupt_id": "int-7", "type": "tool_approval"},
            },
        }

    async def fake_resume(**_kwargs):
        return {
            "child_run_id": "child-7",
            "status": "completed",
            "final_answer": "Web result",
            "result": {"status": "completed", "answer": "Web result", "final_answer": "Web result"},
        }

    monkeypatch.setattr("langgraph_runtime.workflows.delegated_child_nodes._interrupt", fake_interrupt)
    monkeypatch.setattr("langgraph_runtime.workflows.auto_nodes.delegate_sync", fake_delegate_sync)
    monkeypatch.setattr(
        "langgraph_runtime.workflows.delegated_child_nodes.delegate_child_resume_sync",
        fake_resume,
    )

    state = {
        "delegate_workflow_id": "router_rag_agent",
        "agent_run_id": "parent-run",
        "thread_id": "thread-1",
        "question": "Search the web",
        "pre_fetch_bundle": {},
        "delegate_invocation_count": 0,
        "node_events": [],
    }
    update = await workflow_delegate_node(state, {"configurable": {"trace_recorder": None}})

    assert interrupt_payload.get("delegated_child_run_id") == "child-7"
    assert interrupt_payload.get("interrupt_id") == "int-7"
    assert update["child_final_answer"] == "Web result"