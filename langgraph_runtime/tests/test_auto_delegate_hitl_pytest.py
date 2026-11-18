"""Auto delegation must pass inherited tool approval settings to the control plane."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from langgraph_runtime.workflows.auto_nodes import workflow_delegate_node


@pytest.mark.asyncio
async def test_workflow_delegate_passes_inherited_tool_settings(monkeypatch):
    captured: dict = {}

    async def fake_delegate_sync(**kwargs):
        captured.update(kwargs)
        return {
            "child_run_id": "child-run-9",
            "final_answer": "Done",
            "result": {"answer": "Done", "document_sources": [], "web_sources": []},
        }

    monkeypatch.setattr("langgraph_runtime.workflows.auto_nodes.delegate_sync", fake_delegate_sync)

    state = {
        "delegate_workflow_id": "router_rag_agent",
        "agent_run_id": "parent-run",
        "thread_id": "thread-1",
        "question": "Find web benchmarks for transformer latency.",
        "use_web_search": True,
        "web_search_mode": "ask",
        "hitl_web_approval": True,
        "browser_access_mode": "off",
        "use_browser_access": False,
        "hitl_policy": {"enabled": True, "tools": {"search_web": {"mode": "ask", "scope": "run"}}},
        "pre_fetch_bundle": {},
        "delegate_invocation_count": 0,
        "node_events": [],
    }
    update = await workflow_delegate_node(state, {"configurable": {"trace_recorder": None}})
    inherited = captured.get("inherited_tool_settings") or {}
    assert inherited.get("web_search_mode") == "ask"
    assert inherited.get("hitl_web_approval") is True
    assert update["delegated_child_runs"][0]["run_id"] == "child-run-9"