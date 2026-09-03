from __future__ import annotations

import uuid

import pytest

from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.db.models_sqlmodel import AgentRun, AgentTaskCommand, AgentWorkflow
from app.services import agent_task_repository as repository
from app.time_utils import utc_now


def _spec() -> dict:
    return next(
        value["spec_json"] for value in load_builtin_workflows()
        if value["builtin_key"] == "deep_research_agent"
    )


@pytest.mark.asyncio
async def test_course_correction_uses_command_outbox_and_cancel_rejects_it(
    test_session_maker,
    sample_thread,
):
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
    task, _ = await repository.create_task(
        thread_id=sample_thread.id,
        project_id=sample_thread.project_id,
        user_id=None,
        workflow_id="deep_research_agent",
        objective="Research the evidence",
        idempotency_key="correction-task",
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
        run_metadata_json={"run_kind": "agent_task", "runtime_started": True},
    )
    async with test_session_maker() as session:
        async with session.begin():
            session.add(run)
    await repository.attach_run(task.id, run)
    task = await repository.get_task(task.id)

    updated, command, duplicate, correction = await repository.submit_course_correction(
        task.id,
        run_id=run.id,
        expected_version=task.version,
        instruction="  Focus on   security controls. ",
        scope="remaining_work",
        idempotency_key="steer-once",
    )
    _, replayed_command, duplicate, replayed_correction = await repository.submit_course_correction(
        task.id,
        run_id=run.id,
        expected_version=task.version,
        instruction="ignored duplicate body",
        scope="remaining_work",
        idempotency_key="steer-once",
    )

    assert duplicate is True
    assert replayed_command.id == command.id
    assert replayed_correction == correction
    assert correction["instruction"] == "Focus on security controls."
    assert command.result_json["delivery_mode"] == "same_run_safe_boundary"
    assert [value["command_id"] for value in await repository.pending_course_corrections(task.id)] == [command.id]
    assert "course_corrections" not in (updated.config_json or {})

    cancelled, _, _ = await repository.apply_command(
        task.id, action="cancel", idempotency_key="cancel", expected_version=updated.version,
    )
    async with test_session_maker() as session:
        stored = await session.get(AgentTaskCommand, command.id)
        assert stored.status == "rejected"
        assert stored.result_json["delivery_state"] == "rejected"
    assert cancelled.status == "cancelling"


@pytest.mark.asyncio
async def test_hermes_linked_correction_preserves_failed_source_run(
    test_session_maker,
    sample_thread,
):
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id="hermes_deep_research_agent",
                name="Hermes Deep Research",
                description="test",
                visibility="builtin",
                is_builtin=True,
                schema_version=1,
                spec_json={"schema_version": 1},
                metadata_json={"version": 1},
            ))
    task, _ = await repository.create_task(
        thread_id=sample_thread.id,
        project_id=sample_thread.project_id,
        user_id=None,
        workflow_id="hermes_deep_research_agent",
        objective="Research with Hermes",
        idempotency_key="hermes-correction-task",
        config={},
    )
    task, _, _ = await repository.apply_command(
        task.id, action="start", idempotency_key="start", expected_version=task.version,
    )
    source = AgentRun(
        id=str(uuid.uuid4()),
        thread_id=task.thread_id,
        workflow_id=task.workflow_id,
        framework="hermes",
        builder_id="hermes_agent",
        resolved_spec_json={"schema_version": 1},
        run_metadata_json={"run_kind": "agent_task", "runtime_started": True},
    )
    async with test_session_maker() as session:
        async with session.begin():
            session.add(source)
    await repository.attach_run(task.id, source)
    task = await repository.get_task(task.id)
    _, command, _, _ = await repository.submit_course_correction(
        task.id,
        run_id=source.id,
        expected_version=task.version,
        instruction="Use the new scope for remaining work.",
        scope="remaining_work",
        idempotency_key="hermes-steer",
    )
    assert command.result_json["delivery_mode"] == "linked_run"

    async with test_session_maker() as session:
        async with session.begin():
            stored_source = await session.get(AgentRun, source.id, with_for_update=True)
            stored_source.status = "failed"
            stored_source.completed_at = utc_now()
    queued = await repository.queue_linked_course_correction(task.id, run_id=source.id)
    replayed_queue = await repository.queue_linked_course_correction(task.id, run_id=source.id)

    async with test_session_maker() as session:
        stored_source = await session.get(AgentRun, source.id)
        assert stored_source.status == "failed"
    assert queued.status == "queued"
    assert replayed_queue.version == queued.version

    linked = AgentRun(
        id=str(uuid.uuid4()),
        thread_id=task.thread_id,
        workflow_id=task.workflow_id,
        framework="hermes",
        builder_id="hermes_agent",
        resolved_spec_json={"schema_version": 1},
        run_metadata_json={"run_kind": "agent_task", "runtime_started": False},
    )
    async with test_session_maker() as session:
        async with session.begin():
            session.add(linked)
    await repository.attach_run(task.id, linked, parent_run_id=source.id)
    await repository.complete_linked_course_corrections(
        task.id, source_run_id=source.id, linked_run_id=linked.id,
    )
    pending = await repository.pending_course_corrections(task.id)
    assert [value["correction_id"] for value in pending] == [
        command.result_json["correction"]["correction_id"]
    ]
    async with test_session_maker() as session:
        stored_command = await session.get(AgentTaskCommand, command.id)
        stored_linked = await session.get(AgentRun, linked.id)
        assert stored_command.result_json["linked_run_id"] == linked.id
        assert stored_command.result_json["delivery_state"] == "linked"
        assert stored_command.status == "accepted"
        assert stored_linked.parent_run_id == source.id


@pytest.mark.asyncio
async def test_legacy_config_corrections_are_backfilled_idempotently(
    test_session_maker,
    sample_thread,
):
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id="legacy-workflow", name="Legacy", description="test",
                visibility="builtin", is_builtin=True, schema_version=1,
                spec_json={"schema_version": 1}, metadata_json={"version": 1},
            ))
    task, _ = await repository.create_task(
        thread_id=sample_thread.id,
        project_id=sample_thread.project_id,
        user_id=None,
        workflow_id="legacy-workflow",
        objective="Legacy correction",
        idempotency_key="legacy-correction-task",
        config={"course_corrections": [{
            "id": "legacy-correction-1",
            "instruction": "Preserve this correction.",
            "scope": "remaining_work",
            "status": "pending",
        }]},
    )

    first = await repository.pending_course_corrections(task.id)
    second = await repository.pending_course_corrections(task.id)
    refreshed = await repository.get_task(task.id)

    assert [value["correction_id"] for value in first] == ["legacy-correction-1"]
    assert [value["command_id"] for value in second] == [first[0]["command_id"]]
    assert "course_corrections" not in refreshed.config_json


@pytest.mark.asyncio
async def test_budget_review_steer_waits_for_one_durable_runtime_delivery(
    test_session_maker,
    sample_thread,
):
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id="budget-steer-workflow",
                name="Budget steer",
                description="test",
                visibility="builtin",
                is_builtin=True,
                schema_version=1,
                spec_json=_spec(),
                metadata_json={"version": 1},
            ))
    task, _ = await repository.create_task(
        thread_id=sample_thread.id,
        project_id=sample_thread.project_id,
        user_id=None,
        workflow_id="budget-steer-workflow",
        objective="Research until the budget boundary",
        idempotency_key="budget-steer-task",
        config={"limits": {"max_model_calls": 2}},
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
        run_metadata_json={"run_kind": "agent_task", "runtime_started": True},
    )
    async with test_session_maker() as session:
        async with session.begin():
            session.add(run)
    await repository.attach_run(task.id, run)
    awaiting, pending = await repository.create_budget_review(
        task.id,
        run_id=run.id,
        provisional_answer="A useful but incomplete answer.",
        warnings=[{"code": "budget_tranche_exhausted"}],
        gaps=["remaining topic"],
    )

    steered, duplicate, linked = await repository.respond_to_budget_review(
        task.id,
        run_id=run.id,
        interrupt_id=pending["interrupt_id"],
        expected_version=awaiting.version,
        decision="steer",
        guidance="Focus the next tranche on the remaining topic.",
        idempotency_key="budget-steer-once",
    )
    command = await repository.get_course_correction_command(
        task.id, idempotency_key="budget-review:budget-steer-once",
    )

    assert duplicate is False and linked is False
    assert steered.current_phase == "budget_correction_delivery_pending"
    assert command is not None
    assert command.result_json["correction"]["instruction"] == "Focus the next tranche on the remaining topic."
    assert (await repository.get_task_run(task.id)).pending_interrupt_json["decision"] == {
        "action": "steer",
        "idempotency_key": "budget-steer-once",
        "guidance_delivery": "course_correction_command",
    }
    assert await repository.claim_next_task("worker-before-runtime-acceptance") is None

    await repository.mark_course_correction_delivered(
        command.id,
        receipt={
            "status": "accepted",
            "run_id": run.id,
            "correction_id": command.result_json["correction"]["correction_id"],
            "operation_id": command.id,
        },
    )
    released = await repository.get_task(task.id)
    claimed = await repository.claim_next_task("worker-after-runtime-acceptance")

    assert released.current_phase == "budget_continuation_queued"
    assert claimed is not None and claimed.id == task.id
    assert len(await repository.pending_course_corrections(task.id)) == 1

    awaiting_again, pending_again = await repository.create_budget_review(
        task.id,
        run_id=run.id,
        provisional_answer="A second incomplete answer.",
        warnings=[{"code": "budget_tranche_exhausted"}],
        gaps=["another remaining topic"],
    )
    await repository.respond_to_budget_review(
        task.id,
        run_id=run.id,
        interrupt_id=pending_again["interrupt_id"],
        expected_version=awaiting_again.version,
        decision="steer",
        guidance="Use a different evidence source.",
        idempotency_key="budget-steer-rejected",
    )
    rejected_command = await repository.get_course_correction_command(
        task.id, idempotency_key="budget-review:budget-steer-rejected",
    )
    await repository.reject_course_correction(
        rejected_command.id,
        error={"code": "runtime_run_identity_mismatch", "retryable": False},
    )
    restored = await repository.get_task(task.id)
    restored_run = await repository.get_task_run(task.id)

    assert restored.status == "awaiting_approval"
    assert restored.current_phase == "budget_review"
    assert restored_run.status == "awaiting_human"
    assert restored_run.pending_interrupt_json["status"] == "pending"
    assert "decision" not in restored_run.pending_interrupt_json
    assert restored_run.pending_interrupt_json["delivery_error"]["code"] == "runtime_run_identity_mismatch"
