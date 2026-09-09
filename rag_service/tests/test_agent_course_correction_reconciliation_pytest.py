from types import SimpleNamespace

import pytest

from app.services import agent_runtime_reconciliation as reconciliation


@pytest.mark.asyncio
async def test_reconciliation_rejects_correction_after_cancellation(monkeypatch):
    command = SimpleNamespace(id="command-1", task_id="task-1", result_json={})
    task = SimpleNamespace(status="cancelled", deletion_requested_at=None)
    rejected = []

    async def no_runs(_self, *, limit):
        return []

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
        no_runs,
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
