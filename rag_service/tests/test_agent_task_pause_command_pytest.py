from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.api import agent_tasks
from runtime_protocol.errors import RuntimeError as AgentRuntimeError


@pytest.mark.asyncio
async def test_pause_runtime_run_not_found_returns_conflict_not_500(monkeypatch):
    task = type("Task", (), {"id": "task-1", "version": 2, "status": "pausing", "config_json": {}})()
    run = type("Run", (), {"id": "run-1"})()
    command = type("Command", (), {"id": "cmd-1"})()

    monkeypatch.setattr(agent_tasks, "_owned_task", AsyncMock(return_value=task))
    monkeypatch.setattr(agent_tasks, "_require_task_capability", AsyncMock())
    monkeypatch.setattr(
        agent_tasks.repository,
        "apply_command",
        AsyncMock(return_value=(task, command, False)),
    )
    monkeypatch.setattr(agent_tasks.repository, "get_task_run", AsyncMock(return_value=run))
    monkeypatch.setattr(
        agent_tasks,
        "request_task_pause",
        AsyncMock(
            side_effect=AgentRuntimeError(
                "runtime_run_not_found",
                "Runtime run not found",
                retryable=False,
            )
        ),
    )
    monkeypatch.setattr(agent_tasks, "_task_payload", AsyncMock(return_value={"id": task.id, "status": task.status}))

    with pytest.raises(HTTPException) as error:
        await agent_tasks.command_agent_task(
            task.id,
            "pause",
            agent_tasks.AgentTaskCommandRequest(expected_version=1),
            thread_id="thread-1",
            idempotency_key="pause-once",
        )

    assert error.value.status_code == 409
    assert error.value.detail["code"] == "runtime_run_not_found"
