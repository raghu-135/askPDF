from __future__ import annotations

import uuid
from datetime import timedelta
from unittest.mock import AsyncMock

import pytest

from app.db.models_sqlmodel import AgentRun, AgentTask, AgentWorkflow
from app.models.deep_research import AgentTaskStatus
from app.services import agent_task_repository as repository
from app.services import agent_run_cancellation as cancellation
from app.services.agent_task_runtime import execute_claimed_task
from app.time_utils import utc_now
from runtime_protocol.errors import RuntimeError as AgentRuntimeError


async def _awaiting_result_review_task(test_session_maker, sample_thread):
    workflow_id = f"cancel-review-{uuid.uuid4()}"
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id=workflow_id,
                name="Cancel review",
                description="cancel review test",
                visibility="builtin",
                is_builtin=True,
                framework="langgraph",
                builder_id="langgraph_graph",
                spec_json={"schema_version": 1, "workflow_id": workflow_id},
            ))
    task, _ = await repository.create_task(
        thread_id=sample_thread.id,
        project_id=sample_thread.project_id,
        user_id=None,
        workflow_id=workflow_id,
        objective="Cancel during review",
        idempotency_key=str(uuid.uuid4()),
        config={"enabled_profiles": ["document_researcher"]},
    )
    run = AgentRun(
        id=str(uuid.uuid4()),
        thread_id=task.thread_id,
        workflow_id=task.workflow_id,
        framework="langgraph",
        builder_id="langgraph_graph",
        resolved_spec_json={"schema_version": 1},
        run_metadata_json={"run_kind": "agent_task"},
    )
    async with test_session_maker() as session:
        async with session.begin():
            session.add(run)
    run = await repository.attach_run(task.id, run)
    async with test_session_maker() as session:
        async with session.begin():
            stored_task = await session.get(AgentTask, task.id)
            stored_run = await session.get(AgentRun, run.id)
            stored_task.status = AgentTaskStatus.AWAITING_APPROVAL.value
            stored_task.current_phase = "awaiting_result_review"
            stored_task.version = stored_task.version + 1
            stored_run.status = "awaiting_human"
            stored_run.run_metadata_json = {
                **(stored_run.run_metadata_json or {}),
                "runtime_started": True,
                "projection": {"status": "applied"},
            }
            stored_run.pending_interrupt_json = {
                "status": "pending",
                "type": "incomplete_result_review",
                "response_operation": "task.result_review.respond",
                "interrupt_id": "review-1",
                "allowed_actions": ["accept", "retry_with_input"],
            }
    task = await repository.get_task(task.id)
    assert task is not None
    return task, await repository.get_task_run(task.id)


@pytest.mark.asyncio
async def test_product_review_cancel_is_local_without_runtime_round_trip(
    test_session_maker, sample_thread,
):
    task, run = await _awaiting_result_review_task(test_session_maker, sample_thread)
    assert run is not None

    task, _, duplicate = await repository.apply_command(
        task.id,
        action="cancel",
        idempotency_key="cancel-review",
        expected_version=task.version,
    )
    stored_run = await repository.get_task_run(task.id)

    assert duplicate is False
    assert task.status == AgentTaskStatus.CANCELLED.value
    assert stored_run is not None
    assert stored_run.status == "cancelled"
    assert stored_run.pending_interrupt_json == {}
    assert stored_run.completed_at is not None


def test_defer_runtime_cancellation_on_run_not_found_until_confirm_timeout():
    task = type("Task", (), {"updated_at": utc_now()})()
    run = type("Run", (), {"run_metadata_json": {"runtime_started": True}})()
    error = AgentRuntimeError("runtime_run_not_found", "missing", retryable=False)
    assert not cancellation.should_abandon_runtime_cancellation(task, run, error=error)


def test_abandon_runtime_cancellation_on_run_not_found_after_confirm_timeout(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_CANCEL_CONFIRM_TIMEOUT_SECONDS", "120")
    task = type("Task", (), {"updated_at": utc_now() - timedelta(seconds=121)})()
    run = type("Run", (), {"run_metadata_json": {"runtime_started": True}})()
    error = AgentRuntimeError("runtime_run_not_found", "missing", retryable=False)
    assert cancellation.should_abandon_runtime_cancellation(task, run, error=error)


def test_abandon_runtime_cancellation_after_confirm_timeout(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_CANCEL_CONFIRM_TIMEOUT_SECONDS", "120")
    task = type("Task", (), {"updated_at": utc_now() - timedelta(seconds=121)})()
    run = type("Run", (), {"run_metadata_json": {"runtime_started": True}})()
    assert cancellation.should_abandon_runtime_cancellation(task, run)


@pytest.mark.asyncio
async def test_execute_claimed_task_defers_cancellation_when_runtime_run_missing(monkeypatch):
    task = type("Task", (), {
        "id": "task-1",
        "status": AgentTaskStatus.CANCELLING.value,
        "updated_at": utc_now(),
    })()
    run = type("Run", (), {
        "id": "run-1",
        "status": "running",
        "run_metadata_json": {"runtime_started": True},
        "pending_interrupt_json": {},
    })()

    monkeypatch.setattr(
        "app.services.agent_task_runtime.tasks.get_task",
        AsyncMock(side_effect=[task, task]),
    )
    monkeypatch.setattr("app.services.agent_task_runtime.tasks.get_task_run", AsyncMock(return_value=run))
    request = AsyncMock(side_effect=AgentRuntimeError("runtime_run_not_found", "missing", retryable=False))
    abandon = AsyncMock()
    finalize = AsyncMock()
    defer = AsyncMock()
    monkeypatch.setattr("app.services.agent_task_runtime.request_task_cancellation", request)
    monkeypatch.setattr("app.services.agent_task_runtime.abandon_pending_cancellation", abandon)
    monkeypatch.setattr("app.services.agent_task_runtime.finalize_product_local_cancellation", finalize)
    monkeypatch.setattr("app.services.agent_task_runtime.tasks.release_task_lease", AsyncMock())
    monkeypatch.setattr("app.services.agent_task_runtime.tasks.defer_task_lease", defer)

    await execute_claimed_task("task-1", "worker-1")

    abandon.assert_not_awaited()
    finalize.assert_not_awaited()
    defer.assert_awaited_once()
    request.assert_awaited_once()
