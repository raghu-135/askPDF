from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.api import agent_tasks as tasks_api


def _event(*, sequence: int, run_id: str, event_type: str, terminal: bool) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"event-{sequence}",
        event_id=f"event-id-{sequence}",
        sequence=sequence,
        event_type=event_type,
        task_id="task-1",
        agent_run_id=run_id,
        todo_id=None,
        subagent_run_id=None,
        artifact_id=None,
        payload_json={},
        created_at=None,
        occurred_at=None,
        terminal=terminal,
        source_metadata_json={},
    )


@pytest.mark.asyncio
async def test_task_event_stream_queries_incrementally_and_stops_after_scoped_terminal(monkeypatch):
    task = SimpleNamespace(id="task-1", active_run_id="run-1")
    calls: list[int] = []

    async def list_task_runs(_task_id):
        return [SimpleNamespace(id="run-1"), SimpleNamespace(id="run-other")]

    async def list_events(_task_id, *, after_sequence):
        calls.append(after_sequence)
        if after_sequence == 0:
            return [
                _event(sequence=1, run_id="run-other", event_type="run.completed", terminal=True),
                _event(sequence=2, run_id="run-1", event_type="run.started", terminal=False),
            ]
        if after_sequence == 2:
            return [_event(sequence=3, run_id="run-1", event_type="run.completed", terminal=True)]
        return []

    async def no_sleep(_seconds):
        return None

    async def owned_task(*_args, **_kwargs):
        return task

    monkeypatch.setattr(tasks_api, "_owned_task", owned_task)
    monkeypatch.setattr(tasks_api.repository, "list_task_runs", list_task_runs)
    monkeypatch.setattr(tasks_api.repository, "list_events", list_events)
    monkeypatch.setattr(tasks_api.asyncio, "sleep", no_sleep)

    response = await tasks_api.stream_agent_task_events(
        "task-1", thread_id="thread-1", after_sequence=0, run_id="run-1", scope="run"
    )
    chunks = [chunk async for chunk in response.body_iterator]
    payloads = [
        json.loads(line.removeprefix("data: "))
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("data:")
    ]

    assert [payload["sequence"] for payload in payloads] == [2, 3]
    assert payloads[-1]["terminal"] is True
    assert calls == [0, 2]
