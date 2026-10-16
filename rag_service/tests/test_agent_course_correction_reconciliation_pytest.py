from types import SimpleNamespace

import pytest

from app.services import agent_runtime_reconciliation as reconciliation
from app.services import agent_task_repository as task_repository


async def _no_reconciliation_runs(_self, *, limit):
    return []


@pytest.mark.asyncio
async def test_reconciliation_rejects_correction_after_cancellation(monkeypatch):
    command = SimpleNamespace(
        id="command-1", task_id="task-1", status="accepted", result_json={},
    )
    task = SimpleNamespace(status="cancelled", deletion_requested_at=None)
    rejected = []

    async def pending_commands(*, limit):
        return [command]

    async def get_task(task_id):
        assert task_id == "task-1"
        return task

    async def reject(command_id, *, error):
        rejected.append((command_id, error))

    monkeypatch.setattr(
        reconciliation.AgentWorkflowRepository,
        "list_runtime_reconciliation_candidates",
        _no_reconciliation_runs,
    )
    monkeypatch.setattr(
        "app.services.agent_task_repository.list_pending_course_correction_commands",
        pending_commands,
    )
    monkeypatch.setattr("app.services.agent_task_repository.get_task", get_task)
    monkeypatch.setattr(
        "app.services.agent_task_repository.reject_course_correction", reject,
    )

    result = await reconciliation.run_runtime_reconciliation(batch_size=5)

    assert result["corrections"] == 1
    assert result["failed"] == 0
    assert rejected == [
        ("command-1", {"code": "course_correction_cancelled", "retryable": False})
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery_state", ["linked", "delivered", "incorporated"])
async def test_reconciliation_does_not_redeliver_consumed_corrections(
    monkeypatch,
    delivery_state,
):
    command = SimpleNamespace(
        id="command-1",
        task_id="task-1",
        status="accepted",
        result_json={"delivery_mode": "linked_run", "delivery_state": delivery_state},
    )
    calls = []

    async def pending_commands(*, limit):
        return [command]

    async def unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("consumed correction must not be processed")

    monkeypatch.setattr(
        reconciliation.AgentWorkflowRepository,
        "list_runtime_reconciliation_candidates",
        _no_reconciliation_runs,
    )
    monkeypatch.setattr(
        "app.services.agent_task_repository.list_pending_course_correction_commands",
        pending_commands,
    )
    monkeypatch.setattr("app.services.agent_task_repository.get_task", unexpected)

    result = await reconciliation.run_runtime_reconciliation(batch_size=5)

    assert result["corrections"] == 0
    assert result["failed"] == 0
    assert calls == []
    assert command.result_json["delivery_state"] == delivery_state


@pytest.mark.asyncio
@pytest.mark.parametrize("task_status", ["recovery_required", "expired"])
async def test_reconciliation_does_not_resurrect_non_executable_tasks(
    monkeypatch,
    task_status,
):
    command = SimpleNamespace(
        id="command-1",
        task_id="task-1",
        status="accepted",
        result_json={"delivery_mode": "linked_run", "delivery_state": "accepted"},
    )
    task = SimpleNamespace(status=task_status, deletion_requested_at=None)
    queued = []

    async def pending_commands(*, limit):
        return [command]

    async def get_task(_task_id):
        return task

    async def unexpected_queue(*args, **kwargs):
        queued.append((args, kwargs))

    monkeypatch.setattr(
        reconciliation.AgentWorkflowRepository,
        "list_runtime_reconciliation_candidates",
        _no_reconciliation_runs,
    )
    monkeypatch.setattr(
        "app.services.agent_task_repository.list_pending_course_correction_commands",
        pending_commands,
    )
    monkeypatch.setattr("app.services.agent_task_repository.get_task", get_task)
    monkeypatch.setattr(
        "app.services.agent_task_repository.queue_linked_course_correction",
        unexpected_queue,
    )

    result = await reconciliation.run_runtime_reconciliation(batch_size=5)

    assert result["corrections"] == 0
    assert result["failed"] == 0
    assert queued == []


@pytest.mark.asyncio
async def test_reconciliation_queues_an_accepted_linked_correction_once(monkeypatch):
    command = SimpleNamespace(
        id="command-1",
        task_id="task-1",
        status="accepted",
        expected_version=1,
        result_json={
            "delivery_mode": "linked_run",
            "delivery_state": "accepted",
            "source_run_id": "run-1",
            "correction": {"correction_id": "correction-1"},
        },
    )
    task = SimpleNamespace(id="task-1", status="running", deletion_requested_at=None)
    run = SimpleNamespace(id="run-1", status="completed")
    calls = []

    async def pending_commands(*, limit):
        return [command] if task_repository.course_correction_needs_delivery(command) else []

    async def get_task(_task_id):
        return task

    async def get_run(_self, _run_id):
        return run

    async def set_delivery_mode(command_id, *, delivery_mode, receipt=None):
        calls.append(("mode", command_id, delivery_mode))
        command.result_json = {**command.result_json, "delivery_mode": delivery_mode}
        return True

    async def queue(task_id, *, run_id):
        calls.append(("queue", task_id, run_id))
        command.result_json = {**command.result_json, "delivery_state": "linked"}
        task.status = "queued"
        task.current_phase = "course_correction_queued"
        return task

    async def ensure(task_id):
        calls.append(("ensure", task_id))

    monkeypatch.setattr(
        reconciliation.AgentWorkflowRepository,
        "list_runtime_reconciliation_candidates",
        _no_reconciliation_runs,
    )
    monkeypatch.setattr(
        reconciliation.AgentWorkflowRepository,
        "get_run",
        get_run,
    )
    monkeypatch.setattr(
        "app.services.agent_task_repository.list_pending_course_correction_commands",
        pending_commands,
    )
    monkeypatch.setattr("app.services.agent_task_repository.get_task", get_task)
    monkeypatch.setattr(
        "app.services.agent_task_repository.set_course_correction_delivery_mode",
        set_delivery_mode,
    )
    monkeypatch.setattr(
        "app.services.agent_task_repository.queue_linked_course_correction",
        queue,
    )
    monkeypatch.setattr("app.services.agent_task_runtime.ensure_task_run", ensure)

    first = await reconciliation.run_runtime_reconciliation(batch_size=5)
    second = await reconciliation.run_runtime_reconciliation(batch_size=5)

    assert calls == [
        ("mode", "command-1", "linked_run"),
        ("queue", "task-1", "run-1"),
        ("ensure", "task-1"),
    ]
    assert first["corrections"] == 1
    assert second["corrections"] == 0
