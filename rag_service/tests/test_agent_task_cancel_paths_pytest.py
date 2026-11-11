from __future__ import annotations

import uuid

import pytest

from app.db import AgentRunStatus
from app.db.models_sqlmodel import AgentRun, AgentWorkflow
from app.product_orchestration.builtin_workflows import load_builtin_workflows
from app.services import agent_task_repository as repository


def _spec() -> dict:
    return next(
        value["spec_json"] for value in load_builtin_workflows()
        if value["builtin_key"] == "deep_research_agent"
    )


async def _seed_deep_research_workflow(test_session_maker) -> None:
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id="deep_research_agent",
                name="Deep Research",
                description="test",
                visibility="builtin",
                is_builtin=True,
                schema_version=1,
                spec_json=_spec(),
                metadata_json={"version": 1},
            ))


@pytest.mark.asyncio
async def test_queued_cancel_closes_attached_run(test_session_maker, sample_thread):
    await _seed_deep_research_workflow(test_session_maker)
    task, _ = await repository.create_task(
        thread_id=sample_thread.id,
        project_id=sample_thread.project_id,
        user_id=None,
        workflow_id="deep_research_agent",
        objective="Cancel while queued",
        idempotency_key="queued-cancel-task",
        config={},
    )
    task, _, _ = await repository.apply_command(
        task.id, action="start", idempotency_key="start", expected_version=task.version,
    )
    run = AgentRun(
        id=str(uuid.uuid4()),
        thread_id=task.thread_id,
        workflow_id=task.workflow_id,
        framework="langgraph",
        builder_id="langgraph_graph",
        resolved_spec_json=_spec(),
        status=AgentRunStatus.RUNNING.value,
        run_metadata_json={"run_kind": "agent_task", "runtime_started": False},
    )
    async with test_session_maker() as session:
        async with session.begin():
            session.add(run)
    await repository.attach_run(task.id, run)
    queued = await repository.get_task(task.id)
    assert queued.status == "queued"

    cancelled, _, _ = await repository.apply_command(
        task.id, action="cancel", idempotency_key="cancel-queued", expected_version=queued.version,
    )
    assert cancelled.status == "cancelled"
    stored_run = await repository.get_task_run(task.id)
    assert stored_run is not None
    assert stored_run.status == "cancelled"


@pytest.mark.asyncio
async def test_running_pre_submit_cancel_persists_scheduled_marker(test_session_maker, sample_thread):
    await _seed_deep_research_workflow(test_session_maker)
    task, _ = await repository.create_task(
        thread_id=sample_thread.id,
        project_id=sample_thread.project_id,
        user_id=None,
        workflow_id="deep_research_agent",
        objective="Cancel before runtime submit",
        idempotency_key="pre-submit-cancel-task",
        config={},
    )
    task, _, _ = await repository.apply_command(
        task.id, action="start", idempotency_key="start", expected_version=task.version,
    )
    run = AgentRun(
        id=str(uuid.uuid4()),
        thread_id=task.thread_id,
        workflow_id=task.workflow_id,
        framework="langgraph",
        builder_id="langgraph_graph",
        resolved_spec_json=_spec(),
        status=AgentRunStatus.RUNNING.value,
        run_metadata_json={"run_kind": "agent_task", "runtime_started": False},
    )
    async with test_session_maker() as session:
        async with session.begin():
            session.add(run)
    await repository.attach_run(task.id, run)
    running = await repository.claim_next_task("worker-1")
    assert running is not None and running.status == "running"

    cancelling, _, _ = await repository.apply_command(
        task.id, action="cancel", idempotency_key="cancel-presubmit", expected_version=running.version,
    )
    assert cancelling.status == "cancelling"
    stored_run = await repository.get_task_run(task.id)
    assert stored_run.run_metadata_json.get("cancel_requested") is True
    assert stored_run.run_metadata_json["cancellation_request"]["runtime_confirmation"] == "scheduled"
