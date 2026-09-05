from __future__ import annotations

import uuid
from dataclasses import replace

import pytest
from sqlalchemy import select

from app.db.models_sqlmodel import (
    AgentRun,
    AgentRuntimeOperation,
    AgentTaskArtifact,
    AgentTaskEvent,
    AgentTaskRuntimeDelta,
    AgentTaskCommand,
    AgentWorkflow,
)
from app.services import agent_task_repository as repository
from app.services.agent_task_budgets import initial_budget_state
from app.services.agent_task_runtime_projection import (
    RuntimeTaskProjectionConflict,
    _merge_budget,
    apply_neutral_task_completion,
    apply_runtime_task_delta,
)
from runtime_protocol.contracts import RuntimeCourseCorrectionOutcome, RuntimePlanChange, TaskOrchestrationDelta


async def _task_and_run(test_session_maker, sample_thread):
    async with test_session_maker() as session:
        async with session.begin():
            session.add(AgentWorkflow(
                id="runtime-projection-workflow",
                name=f"Runtime projection {uuid.uuid4()}",
                description="projection test",
                visibility="builtin",
                is_builtin=True,
                framework="langgraph",
                builder_id="langgraph_graph",
                spec_json={"schema_version": 1, "workflow_id": "runtime-projection-workflow"},
            ))
    task, _ = await repository.create_task(
        thread_id=sample_thread.id,
        project_id=sample_thread.project_id,
        user_id=None,
        workflow_id="runtime-projection-workflow",
        objective="Research safely",
        idempotency_key=str(uuid.uuid4()),
        config={
            "enabled_profiles": ["document_researcher"],
            "limits": {"max_model_calls": 10},
        },
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
    task = await repository.get_task(task.id)
    assert task is not None
    return task, run


@pytest.mark.asyncio
async def test_projection_recovery_task_is_reserved_for_reconciler(
    test_session_maker, sample_thread,
):
    task, _ = await _task_and_run(test_session_maker, sample_thread)
    await repository.set_task_runtime_status(
        task.id,
        "running",
        phase="runtime_projection_recovery_required",
        reason="runtime_task_projection_conflict",
    )

    assert await repository.claim_next_task("ordinary-task-worker") is None


def _plan(*todo_ids: str) -> dict:
    return {
        "objective": "Research safely",
        "success_criteria": ["Produce an evidence-backed answer"],
        "todos": [
            {
                "id": todo_id,
                "title": f"Research {todo_id}",
                "description": f"Collect evidence for {todo_id}",
                "completion_criteria": f"Evidence for {todo_id} is recorded",
                "profile_id": "document_researcher",
            }
            for todo_id in todo_ids
        ],
    }


def _delta(
    *, run_id: str, operation_id: str, event_id: str, task_version: int,
    plan_revision: int, plan: dict | None, todos: tuple[dict, ...],
    budget_calls: int, interrupt: dict,
) -> TaskOrchestrationDelta:
    budget = initial_budget_state({"max_model_calls": 10})
    budget["tranche_usage"]["model_calls"] = budget_calls
    budget["lifetime_usage"]["model_calls"] = budget_calls
    return TaskOrchestrationDelta(
        event_id=event_id,
        attempt_id=f"{run_id}:attempt:1",
        operation_id=operation_id,
        idempotency_key=f"delta:{operation_id}:{event_id}",
        observed_task_version=task_version,
        observed_plan_revision=plan_revision,
        plan_changes=(RuntimePlanChange(
            runtime_revision=plan_revision + 1,
            parent_runtime_revision=plan_revision,
            acknowledged_product_revision=plan_revision,
            reason="runtime_projection",
            planner_visit=plan_revision + 1,
            plan=plan,
        ),) if plan is not None else (),
        todo_changes=todos,
        subagent_changes=tuple(
            {
                "todo_id": value["id"],
                "profile_id": "document_researcher",
                "attempt": 1,
                "status": "completed",
            }
            for value in todos
            if value.get("status") == "completed"
        ),
        budget_usage=budget,
        pending_interrupt=interrupt,
        result={"status": "awaiting_human" if interrupt.get("operation") == "set" else "completed"},
    )


def test_budget_projection_merges_late_cumulative_snapshots_monotonically():
    current = initial_budget_state({"max_model_calls": 10})
    current["tranche_usage"]["elapsed_active_ms"] = 1_000
    current["lifetime_usage"]["elapsed_active_ms"] = 1_000
    incoming = initial_budget_state({"max_model_calls": 10})
    incoming["tranche_usage"]["elapsed_active_ms"] = 400
    incoming["lifetime_usage"]["elapsed_active_ms"] = 400

    merged = _merge_budget(current, incoming, {"max_model_calls": 10})

    assert merged["tranche_usage"]["elapsed_active_ms"] == 1_000
    assert merged["lifetime_usage"]["elapsed_active_ms"] == 1_000


def test_budget_projection_resets_tranche_but_preserves_lifetime_snapshot():
    current = initial_budget_state({"max_model_calls": 10})
    current["tranche_index"] = 1
    current["tranche_usage"]["elapsed_active_ms"] = 1_000
    current["lifetime_usage"]["elapsed_active_ms"] = 1_000
    incoming = initial_budget_state({"max_model_calls": 10})
    incoming["tranche_index"] = 2
    incoming["tranche_usage"]["elapsed_active_ms"] = 100
    incoming["lifetime_usage"]["elapsed_active_ms"] = 1_100

    merged = _merge_budget(
        current,
        incoming,
        {"max_model_calls": 10},
        authorized_tranche_increment=True,
    )

    assert merged["tranche_usage"]["elapsed_active_ms"] == 100
    assert merged["lifetime_usage"]["elapsed_active_ms"] == 1_100


@pytest.mark.asyncio
async def test_projection_applies_multiple_boundaries_and_exact_replay(
    test_session_maker, sample_thread,
):
    task, run = await _task_and_run(test_session_maker, sample_thread)
    first = _delta(
        run_id=run.id,
        operation_id="start-command",
        event_id=f"{run.id}:attempt:1:operation:start-command:result",
        task_version=task.version,
        plan_revision=0,
        plan=_plan("evidence-a"),
        todos=({"id": "evidence-a", "status": "completed", "progress": 100},),
        budget_calls=1,
        interrupt={
            "operation": "set",
            "value": {
                "interrupt_id": "review-1",
                "type": "budget_review",
                "status": "pending",
                "response_operation": "task.budget_review.respond",
                "allowed_actions": ["continue", "steer", "accept_partial"],
            },
        },
    )
    await apply_runtime_task_delta(task_id=task.id, agent_run_id=run.id, delta=first)
    await apply_runtime_task_delta(task_id=task.id, agent_run_id=run.id, delta=first)

    task = await repository.get_task(task.id)
    assert task is not None
    second = _delta(
        run_id=run.id,
        operation_id="resume-command",
        event_id=f"{run.id}:attempt:1:operation:resume-command:result",
        task_version=task.version,
        plan_revision=1,
        plan=_plan("evidence-a", "evidence-b"),
        todos=(
            {"id": "evidence-a", "status": "running", "progress": 5},
            {"id": "evidence-b", "status": "pending", "progress": 0},
        ),
        budget_calls=2,
        interrupt={"operation": "clear"},
    )
    await apply_runtime_task_delta(task_id=task.id, agent_run_id=run.id, delta=second)

    todos = {value.id: value for value in await repository.list_todos(task.id)}
    refreshed = await repository.get_task(task.id)
    async with test_session_maker() as session:
        ledger = list((await session.execute(
            select(AgentTaskRuntimeDelta).where(AgentTaskRuntimeDelta.task_id == task.id)
        )).scalars().all())
        stored_run = await session.get(AgentRun, run.id)
    assert len(ledger) == 2
    assert todos["evidence-a"].status == "completed"
    assert todos["evidence-b"].status == "pending"
    assert refreshed is not None
    assert refreshed.budgets_json["lifetime_usage"]["model_calls"] == 2
    assert stored_run is not None and stored_run.pending_interrupt_json == {}


@pytest.mark.asyncio
async def test_projection_rejects_conflicting_identity_without_partial_state(
    test_session_maker, sample_thread,
):
    task, run = await _task_and_run(test_session_maker, sample_thread)
    event_id = f"{run.id}:attempt:1:operation:start-command:result"
    first = _delta(
        run_id=run.id,
        operation_id="start-command",
        event_id=event_id,
        task_version=task.version,
        plan_revision=0,
        plan=_plan("evidence-a"),
        todos=({"id": "evidence-a", "status": "pending"},),
        budget_calls=1,
        interrupt={"operation": "clear"},
    )
    await apply_runtime_task_delta(task_id=task.id, agent_run_id=run.id, delta=first)
    conflicting = replace(first, todo_changes=({"id": "evidence-a", "status": "completed"},))
    with pytest.raises(RuntimeTaskProjectionConflict, match="identity was reused"):
        await apply_runtime_task_delta(task_id=task.id, agent_run_id=run.id, delta=conflicting)

    todos = await repository.list_todos(task.id)
    async with test_session_maker() as session:
        ledger = list((await session.execute(
            select(AgentTaskRuntimeDelta).where(AgentTaskRuntimeDelta.task_id == task.id)
        )).scalars().all())
    assert len(ledger) == 1
    assert todos[0].status == "pending"


@pytest.mark.asyncio
async def test_warning_result_enters_review_in_the_delta_transaction(
    test_session_maker, sample_thread,
):
    task, run = await _task_and_run(test_session_maker, sample_thread)
    delta = _delta(
        run_id=run.id,
        operation_id="warning-result",
        event_id=f"{run.id}:attempt:1:operation:warning-result:result",
        task_version=task.version,
        plan_revision=0,
        plan=_plan("evidence-a"),
        todos=({"id": "evidence-a", "status": "completed", "progress": 100},),
        budget_calls=1,
        interrupt={"operation": "clear"},
    )
    delta = replace(delta, result={
        "status": "completed",
        "result_outcome": "completed_with_warnings",
        "warnings": [{"code": "evidence_critic_issues"}],
        "incomplete_reasons": ["A source remains unverified."],
        "task_result": {
            "status": "completed_with_warnings",
            "text": "A useful provisional report.",
            "warnings": [{"code": "evidence_critic_issues"}],
            "gaps": ["A source remains unverified."],
            "usage": {},
        },
    })

    await apply_runtime_task_delta(task_id=task.id, agent_run_id=run.id, delta=delta)

    refreshed = await repository.get_task(task.id)
    stored_run = await repository.get_task_run(task.id)
    events = await repository.list_events(task.id, agent_run_id=run.id)
    assert refreshed is not None and refreshed.status == "awaiting_approval"
    assert stored_run is not None and stored_run.status == "awaiting_human"
    assert stored_run.pending_interrupt_json["type"] == "incomplete_result_review"
    assert stored_run.pending_interrupt_json["allowed_actions"] == ["accept", "retry_with_input"]
    assert sum(event.event_type == "runtime.event" and (event.source_metadata_json or {}).get("source_event") == "task.result_review_requested" for event in events) == 1
    assert not any(event.terminal for event in events)


@pytest.mark.asyncio
async def test_hermes_usage_snapshot_is_applied_exactly_once(
    test_session_maker, sample_thread,
):
    task, run = await _task_and_run(test_session_maker, sample_thread)
    operation_id = "hermes-operation-1"
    usage = {
        "operation_id": operation_id,
        "model_tokens": 93_422,
        "model_calls": None,
        "tool_calls": 4,
        "active_runtime_ms": 165_223,
        "measured_dimensions": ["model_tokens", "tool_calls", "active_runtime_ms"],
    }

    task_result = {
        "status": "completed", "text": "Hermes completed answer.",
        "warnings": [], "gaps": [], "usage": usage,
    }
    await apply_neutral_task_completion(
        task_id=task.id, agent_run_id=run.id, operation_id=operation_id,
        runtime_status="completed", task_result=task_result,
    )
    await apply_neutral_task_completion(
        task_id=task.id, agent_run_id=run.id, operation_id=operation_id,
        runtime_status="completed", task_result=task_result,
    )

    refreshed = await repository.get_task(task.id)
    assert refreshed is not None
    assert refreshed.status == "completed"
    assert refreshed.budgets_json["lifetime_usage"]["model_tokens"] == 93_422
    assert refreshed.budgets_json["lifetime_usage"]["tool_calls"] == 4
    assert refreshed.budgets_json["lifetime_usage"]["elapsed_active_ms"] == 165_223
    assert refreshed.budgets_json["lifetime_usage"]["model_calls"] == 0
    async with test_session_maker() as session:
        records = list((await session.execute(select(AgentRuntimeOperation).where(
            AgentRuntimeOperation.run_id == run.id,
            AgentRuntimeOperation.operation == "task.completion.project",
        ))).scalars().all())
        terminal_events = list((await session.execute(select(AgentTaskEvent).where(
            AgentTaskEvent.task_id == task.id,
            AgentTaskEvent.agent_run_id == run.id,
            AgentTaskEvent.terminal.is_(True),
        ))).scalars().all())
    assert len(records) == 1
    assert records[0].result_json["usage_fingerprint"]
    assert len(terminal_events) == 1


@pytest.mark.asyncio
async def test_projection_translates_artifacts_and_accounts_for_bytes(
    test_session_maker, sample_thread,
):
    task, run = await _task_and_run(test_session_maker, sample_thread)
    initial = _delta(
        run_id=run.id,
        operation_id="start-command",
        event_id=f"{run.id}:attempt:1:operation:start-command:result",
        task_version=task.version,
        plan_revision=0,
        plan=_plan("evidence-a"),
        todos=({"id": "evidence-a", "status": "pending"},),
        budget_calls=1,
        interrupt={"operation": "clear"},
    )
    await apply_runtime_task_delta(task_id=task.id, agent_run_id=run.id, delta=initial)
    task = await repository.get_task(task.id)
    assert task is not None
    content = "bounded runtime evidence"
    runtime_artifact_id = "runtime:artifact-1"
    budget = initial_budget_state({"max_model_calls": 10})
    budget["tranche_usage"]["model_calls"] = 1
    budget["lifetime_usage"]["model_calls"] = 1
    budget["lifetime_usage"]["artifact_bytes"] = len(content.encode())
    delta = TaskOrchestrationDelta(
        event_id=f"{run.id}:attempt:1:operation:continue-command:result",
        attempt_id=f"{run.id}:attempt:1",
        operation_id="continue-command",
        idempotency_key="delta:continue-command:artifact",
        observed_task_version=task.version,
        observed_plan_revision=1,
        todo_changes=({
            "id": "evidence-a",
            "status": "completed",
            "progress": 100,
            "artifact_ids": [runtime_artifact_id],
        },),
        artifacts=({
            "artifact_id": runtime_artifact_id,
            "kind": "intermediate_report",
            "content": content,
            "todo_id": "evidence-a",
            "media_type": "text/plain",
        },),
        budget_usage=budget,
        pending_interrupt={"operation": "clear"},
        result={"status": "completed", "final_artifact_id": runtime_artifact_id},
    )
    artifact_map = await apply_runtime_task_delta(task_id=task.id, agent_run_id=run.id, delta=delta)

    product_artifact_id = artifact_map[runtime_artifact_id]
    todos = await repository.list_todos(task.id)
    refreshed = await repository.get_task(task.id)
    async with test_session_maker() as session:
        artifact = await session.get(AgentTaskArtifact, product_artifact_id)
        stored_run = await session.get(AgentRun, run.id)
    assert artifact is not None and artifact.sha256
    assert todos[0].artifact_ids_json == [product_artifact_id]
    assert refreshed is not None
    assert refreshed.budgets_json["lifetime_usage"]["artifact_bytes"] == len(content.encode())
    assert stored_run is not None
    assert stored_run.run_metadata_json["orchestration_result"]["final_artifact_id"] == product_artifact_id


@pytest.mark.asyncio
async def test_redirect_projects_intermediate_plan_before_retained_artifact(
    test_session_maker, sample_thread,
):
    task, run = await _task_and_run(test_session_maker, sample_thread)
    initial = _delta(
        run_id=run.id, operation_id="start", event_id=f"{run.id}:attempt:1:operation:start:result",
        task_version=task.version, plan_revision=0, plan=_plan("T1"),
        todos=({"id": "T1", "status": "completed", "progress": 100},),
        budget_calls=1, interrupt={"operation": "clear"},
    )
    await apply_runtime_task_delta(task_id=task.id, agent_run_id=run.id, delta=initial)
    task = await repository.get_task(task.id)
    assert task is not None
    correction_id = str(uuid.uuid4())
    command = AgentTaskCommand(
        task_id=task.id, action="steer", idempotency_key="redirect-projection",
        expected_version=task.version, status="accepted",
        result_json={
            "correction": {
                "correction_id": correction_id, "id": correction_id,
                "operation_id": "redirect-command",
                "instruction": "Focus on open-source security controls.",
                "scope": "remaining_work", "status": "accepted",
            },
            "delivery_mode": "same_run_safe_boundary", "delivery_state": "accepted",
            "source_run_id": run.id,
        },
    )
    async with test_session_maker() as session:
        async with session.begin():
            session.add(command)
    content = "research completed before redirect"
    budget = initial_budget_state({"max_model_calls": 10})
    budget["tranche_usage"]["model_calls"] = 3
    budget["lifetime_usage"]["model_calls"] = 3
    budget["lifetime_usage"]["artifact_bytes"] = len(content.encode())
    redirected = TaskOrchestrationDelta(
        event_id=f"{run.id}:attempt:1:operation:redirect:result",
        attempt_id=f"{run.id}:attempt:1", operation_id="redirect",
        idempotency_key="delta:redirect:lineage", observed_task_version=task.version,
        observed_plan_revision=1,
        plan_changes=(
            RuntimePlanChange(
                runtime_revision=2, parent_runtime_revision=1,
                acknowledged_product_revision=1, reason="bounded_replan", planner_visit=2,
                plan=_plan("T2"),
            ),
            RuntimePlanChange(
                runtime_revision=3, parent_runtime_revision=2,
                acknowledged_product_revision=1, reason="course_correction", planner_visit=3,
                plan=_plan("redirected-open-source"), correction_ids=(correction_id,),
            ),
        ),
        todo_changes=(
            {"id": "T1", "status": "completed", "progress": 100},
            {"id": "T2", "status": "completed", "progress": 100, "artifact_ids": ["runtime:T2"]},
            {"id": "redirected-open-source", "status": "completed", "progress": 100},
        ),
        budget_usage=budget,
        artifacts=({
            "artifact_id": "runtime:T2", "kind": "intermediate_report", "content": content,
            "todo_id": "T2", "media_type": "text/plain",
        },),
        pending_interrupt={"operation": "clear"}, result={"status": "completed"},
        correction_outcomes=(RuntimeCourseCorrectionOutcome(
            correction_id=correction_id, operation_id=command.id, state="satisfied",
            runtime_plan_revision=3, todo_ids=("redirected-open-source",),
            explanation="The redirected security comparison was completed.",
        ),),
    )

    await apply_runtime_task_delta(task_id=task.id, agent_run_id=run.id, delta=redirected)

    plans = await repository.list_plans(task.id)
    todos = {value.id: value for value in await repository.list_todos(task.id)}
    assert [value.revision for value in plans] == [1, 2, 3]
    assert todos["T2"].status == "completed"
    assert todos["T2"].artifact_ids_json
    async with test_session_maker() as session:
        stored_command = await session.get(AgentTaskCommand, command.id)
    assert stored_command.status == "completed"
    assert stored_command.result_json["delivery_state"] == "satisfied"
