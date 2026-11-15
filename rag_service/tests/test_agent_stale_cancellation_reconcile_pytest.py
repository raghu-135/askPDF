from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.deep_research import AgentTaskStatus
from app.services import agent_run_cancellation as cancellation
from app.time_utils import utc_now


def _task(**overrides):
    base = {
        "id": "task-1",
        "status": AgentTaskStatus.CANCELLING.value,
        "updated_at": utc_now() - timedelta(seconds=300),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _run(**overrides):
    base = {
        "id": "run-1",
        "thread_id": "thread-1",
        "workflow_id": "workflow-1",
        "framework": "langgraph",
        "builder_id": "langgraph_graph",
        "definition_category": "task",
        "runtime_binding_json": {},
        "runtime_binding_status": "active",
        "status": "running",
        "task_id": "task-1",
        "resolved_spec_json": {},
        "run_metadata_json": {
            "runtime_started": True,
            "cancellation_request": {
                "requested_at": (utc_now() - timedelta(seconds=300)).isoformat(),
            },
        },
        "metrics_json": {},
        "debug_trace_json": {},
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.asyncio
async def test_fresh_cancelling_task_is_not_reconciled(monkeypatch):
    task = _task(updated_at=utc_now())
    run = _run(run_metadata_json={
        "runtime_started": True,
        "cancellation_request": {"requested_at": utc_now().isoformat()},
    })
    monkeypatch.setattr("app.services.agent_task_repository.get_task", AsyncMock(return_value=task))

    assert await cancellation.reconcile_stale_cancelling_task(task, run) == "fresh"


@pytest.mark.asyncio
async def test_stale_cancelling_without_run_finalizes_task(monkeypatch):
    task = _task()
    complete = AsyncMock()
    monkeypatch.setattr("app.services.agent_task_repository.get_task", AsyncMock(return_value=task))
    monkeypatch.setattr("app.services.agent_task_repository.get_task_run", AsyncMock(return_value=None))
    monkeypatch.setattr("app.services.agent_task_repository.complete_task", complete)

    assert await cancellation.reconcile_stale_cancelling_task(task, None) == "finalized"
    complete.assert_awaited_once_with("task-1", status=AgentTaskStatus.CANCELLED.value, reason="cancelled_by_user")


@pytest.mark.asyncio
async def test_stale_cancelling_abandons_when_runtime_has_no_live_work(monkeypatch):
    task = _task()
    run = _run()
    abandon = AsyncMock()
    adapter = SimpleNamespace(
        inspect_state=AsyncMock(return_value={
            "status": "cancelled",
            "result": {"status": "cancelled"},
            "terminal_event_id": "terminal-1",
            "continuation_available": False,
        })
    )

    monkeypatch.setattr("app.services.agent_task_repository.get_task", AsyncMock(return_value=task))
    monkeypatch.setattr(cancellation, "get_runtime_registry", lambda: SimpleNamespace(get=lambda _definition: adapter))
    monkeypatch.setattr(cancellation, "confirm_task_cancellation", abandon)

    assert await cancellation.reconcile_stale_cancelling_task(task, run) == "finalized"
    abandon.assert_awaited_once()


@pytest.mark.asyncio
async def test_stale_cancelling_defers_when_runtime_still_finishing(monkeypatch):
    task = _task()
    run = _run()
    abandon = AsyncMock()
    adapter = SimpleNamespace(
        inspect_state=AsyncMock(return_value={
            "status": "running",
            "continuation_available": True,
        })
    )

    monkeypatch.setattr("app.services.agent_task_repository.get_task", AsyncMock(return_value=task))
    monkeypatch.setattr(cancellation, "get_runtime_registry", lambda: SimpleNamespace(get=lambda _definition: adapter))
    monkeypatch.setattr(cancellation, "abandon_pending_cancellation", abandon)

    assert await cancellation.reconcile_stale_cancelling_task(task, run) == "deferred"
    abandon.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_is_idempotent_after_cancelled(monkeypatch):
    task = _task(status=AgentTaskStatus.CANCELLED.value)
    run = _run(status="cancelled")
    monkeypatch.setattr("app.services.agent_task_repository.get_task", AsyncMock(return_value=task))

    assert await cancellation.reconcile_stale_cancelling_task(task, run) == "not_required"


@pytest.mark.asyncio
async def test_terminal_run_mismatch_reconciled_even_when_fresh(monkeypatch):
    task = _task(updated_at=utc_now())
    run = _run(
        status="completed",
        run_metadata_json={
            "runtime_started": True,
            "projection": {"runtime_result": {"status": "completed", "answer": "done"}},
            "cancellation_request": {"requested_at": utc_now().isoformat()},
        },
    )
    reconcile = AsyncMock(return_value="projected")

    monkeypatch.setattr("app.services.agent_task_repository.get_task", AsyncMock(side_effect=[
        task,
        SimpleNamespace(id="task-1", status=AgentTaskStatus.COMPLETED.value),
    ]))
    monkeypatch.setattr("app.services.agent_runtime_reconciliation.reconcile_run_by_id", reconcile)

    assert await cancellation.reconcile_stale_cancelling_task(task, run) == "projected"
    reconcile.assert_awaited_once_with("run-1")


@pytest.mark.asyncio
async def test_bounded_stale_reconciliation_reports_outcomes(monkeypatch):
    tasks = [_task(id="task-1"), _task(id="task-2")]

    async def list_stale(*, limit):
        assert limit == 2
        return tasks

    async def reconcile(task, run):
        return "finalized" if task.id == "task-1" else "deferred"

    monkeypatch.setattr("app.services.agent_task_repository.list_stale_cancelling_tasks", list_stale)
    monkeypatch.setattr("app.services.agent_task_repository.get_task_run", AsyncMock(return_value=_run()))
    monkeypatch.setattr(cancellation, "reconcile_stale_cancelling_task", reconcile)

    assert await cancellation.run_stale_cancellation_reconciliation(batch_size=2) == {
        "inspected": 2,
        "fresh": 0,
        "finalized": 1,
        "projected": 0,
        "deferred": 1,
        "not_required": 0,
        "missing": 0,
        "failed": 0,
    }