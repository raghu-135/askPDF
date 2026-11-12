from __future__ import annotations

import pytest

from app.db.enums import AgentRunStatus
from app.db.jsonb_utils import replace_jsonb_field
from app.models.deep_research import AgentTaskStatus
from app.services import agent_task_repository as repository
from app.services.agent_task_repository import AgentTaskConflict
from test_agent_task_runtime_projection_pytest import _task_and_run


@pytest.mark.asyncio
async def test_align_product_owned_review_state_repairs_running_task(test_session_maker, sample_thread):
    task, run = await _task_and_run(test_session_maker, sample_thread)
    async with test_session_maker() as session:
        async with session.begin():
            from app.db.models_sqlmodel import AgentRun, AgentTask

            stored_task = await session.get(AgentTask, task.id)
            stored_run = await session.get(AgentRun, run.id)
            stored_task.status = AgentTaskStatus.RUNNING.value
            stored_run.status = AgentRunStatus.RUNNING.value
            replace_jsonb_field(stored_run, "pending_interrupt_json", {
                "interrupt_id": "budget-review:run",
                "type": "budget_review",
                "response_operation": "task.budget_review.respond",
                "status": "pending",
                "allowed_actions": ["continue", "steer"],
            })

    aligned = await repository.align_product_owned_review_state(task.id, run.id)

    assert aligned.status == AgentTaskStatus.AWAITING_APPROVAL.value
    refreshed_run = await repository.get_task_run(task.id)
    assert refreshed_run is not None
    assert refreshed_run.status == AgentRunStatus.AWAITING_HUMAN.value


@pytest.mark.asyncio
async def test_budget_review_respond_after_alignment(test_session_maker, sample_thread):
    task, run = await _task_and_run(test_session_maker, sample_thread)
    async with test_session_maker() as session:
        async with session.begin():
            from app.db.models_sqlmodel import AgentRun, AgentTask

            stored_task = await session.get(AgentTask, task.id)
            stored_run = await session.get(AgentRun, run.id)
            stored_task.status = AgentTaskStatus.RUNNING.value
            stored_run.status = AgentRunStatus.RUNNING.value
            replace_jsonb_field(stored_run, "pending_interrupt_json", {
                "interrupt_id": "budget-review:run",
                "type": "budget_review",
                "response_operation": "task.budget_review.respond",
                "status": "pending",
                "allowed_actions": ["continue", "steer"],
                "provisional_answer": "Partial answer",
            })

    await repository.align_product_owned_review_state(task.id, run.id)
    task = await repository.get_task(task.id)
    run = await repository.get_task_run(task.id)
    pending = run.pending_interrupt_json
    updated, duplicate, linked = await repository.respond_to_budget_review(
        task.id,
        run_id=run.id,
        interrupt_id=pending["interrupt_id"],
        expected_version=task.version,
        decision="continue",
        guidance=None,
        idempotency_key="align-continue",
    )
    assert duplicate is False
    assert updated.status == AgentTaskStatus.QUEUED.value


@pytest.mark.asyncio
async def test_budget_review_still_rejects_without_pending(test_session_maker, sample_thread):
    task, run = await _task_and_run(test_session_maker, sample_thread)

    with pytest.raises(AgentTaskConflict) as conflict:
        await repository.respond_to_budget_review(
            task.id,
            run_id=run.id,
            interrupt_id="missing",
            expected_version=task.version,
            decision="continue",
            guidance=None,
            idempotency_key="missing",
        )
    assert conflict.value.code in {
        "budget_review_not_pending",
        "budget_review_identity_mismatch",
    }
