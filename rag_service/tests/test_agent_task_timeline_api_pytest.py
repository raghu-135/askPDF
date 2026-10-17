from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api import agent_tasks as agent_tasks_api


@pytest.mark.asyncio
async def test_timeline_api_emits_research_guidance_without_parent_run_id(monkeypatch):
    task = SimpleNamespace(
        id="task-1",
        objective="Compare bakery metrics",
        config_json={
            "result_review_context": [
                {
                    "source_run_id": "run-1",
                    "followup_input": "Focus on the PDF only.",
                    "idempotency_key": "review-key-1",
                },
            ],
        },
        version=3,
        thread_id="thread-1",
        workflow_id="wf-1",
        status="running",
        budgets_json={},
    )
    run = SimpleNamespace(
        id="run-2",
        task_id="task-1",
        parent_run_id=None,
        task_attempt=2,
        runtime_binding_status=None,
        metrics_json={},
        started_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        completed_at=None,
        status="running",
        pending_interrupt_json=None,
        run_metadata_json={
            "result_review_source_run_id": "run-1",
            "result_review_idempotency_key": "review-key-1",
        },
        error_json=None,
        debug_trace_json=None,
    )
    plan = SimpleNamespace(
        id="plan-1",
        plan_json={"todos": []},
        objective="Plan",
        reason="initial",
        revision=1,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        completion_criteria_json=[],
    )

    monkeypatch.setattr(agent_tasks_api, "_owned_task", AsyncMock(return_value=task))
    monkeypatch.setattr(agent_tasks_api.repository, "list_task_runs", AsyncMock(return_value=[run]))
    monkeypatch.setattr(agent_tasks_api.repository, "list_plans", AsyncMock(return_value=[plan]))
    monkeypatch.setattr(agent_tasks_api.repository, "list_todos", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent_tasks_api.repository, "list_subagent_runs", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent_tasks_api.repository, "list_artifacts", AsyncMock(return_value=[]))
    monkeypatch.setattr(agent_tasks_api.repository, "list_events", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        agent_tasks_api,
        "_task_payload",
        AsyncMock(return_value={"id": task.id, "research_canvases": []}),
    )
    monkeypatch.setattr(agent_tasks_api, "attach_research_canvases", AsyncMock())

    payload = await agent_tasks_api.get_agent_task_timeline("task-1", "run-2", thread_id="thread-1")
    guidance = [item for item in payload["items"] if item["type"] == "research_guidance"]
    assert len(guidance) == 1
    assert guidance[0]["primary_content"] == "Focus on the PDF only."
