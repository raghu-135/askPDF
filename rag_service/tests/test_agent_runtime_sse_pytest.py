from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.api import agent_workflows as api


@pytest.mark.asyncio
async def test_product_run_event_stream_closes_after_terminal_event(monkeypatch):
    run = SimpleNamespace(id="run-1", thread_id="thread-1")

    class Repository:
        async def list_run_events(self, _run_id):
            return [
                SimpleNamespace(
                    id="event-1",
                    event_id="event-1",
                    sequence=1,
                    attempt=1,
                    kind="run.completed",
                    payload_json={"status": "completed"},
                    occurred_at=None,
                    created_at=None,
                )
            ]

    async def owned_run(_run_id, _thread_id):
        return run

    monkeypatch.setattr(api, "_owned_run_for_operation", owned_run)
    monkeypatch.setattr(api, "AgentWorkflowRepository", Repository)
    request = SimpleNamespace(is_disconnected=lambda: _false())
    response = await api.stream_agent_run_events("run-1", request=request, thread_id="thread-1", after_sequence=0)

    chunks = [chunk async for chunk in response.body_iterator]
    payloads = [
        json.loads(line.removeprefix("data: "))
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("data:")
    ]

    assert len(payloads) == 1
    assert payloads[0]["terminal"] is True


@pytest.mark.asyncio
async def test_product_run_event_stream_disambiguates_duplicate_persisted_event_ids(monkeypatch):
    run = SimpleNamespace(id="run-1", thread_id="thread-1")
    rows = [
        SimpleNamespace(
            id="journal-1", event_id="runtime-event-1", sequence=1, attempt=1,
            kind="dispatch.started",
            payload_json={"dispatch_id": "dispatch-1", "dispatch_mode": "parallel", "planned": 1},
            occurred_at=None, created_at=None,
        ),
        SimpleNamespace(
            id="journal-2", event_id="runtime-event-1", sequence=2, attempt=1,
            kind="worker.completed",
            payload_json={"dispatch_id": "dispatch-1", "dispatch_mode": "parallel", "work_id": "work-1"},
            occurred_at=None, created_at=None,
        ),
        SimpleNamespace(
            id="journal-3", event_id="runtime-event-2", sequence=3, attempt=1,
            kind="run.completed", payload_json={"status": "completed"},
            occurred_at=None, created_at=None,
        ),
    ]

    class Repository:
        async def list_run_events(self, _run_id):
            return rows

    async def owned_run(_run_id, _thread_id):
        return run

    monkeypatch.setattr(api, "_owned_run_for_operation", owned_run)
    monkeypatch.setattr(api, "AgentWorkflowRepository", Repository)
    response = await api.stream_agent_run_events(
        "run-1", request=SimpleNamespace(is_disconnected=lambda: _false()),
        thread_id="thread-1", after_sequence=0,
    )

    chunks = [chunk async for chunk in response.body_iterator]
    payloads = [
        json.loads(line.removeprefix("data: "))
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("data:")
    ]
    assert [payload["event_id"] for payload in payloads] == [
        "runtime-event-1:journal:journal-1",
        "runtime-event-1:journal:journal-2",
        "runtime-event-2",
    ]
    group = next(group for group in payloads[1]["parallel_groups"] if group["group_id"] == "dispatch-1")
    assert set(group["event_ids"]) == {
        "runtime-event-1:journal:journal-1",
        "runtime-event-1:journal:journal-2",
    }
    assert all(
        event_id in {payload["event_id"] for payload in payloads}
        for event_id in group["event_ids"]
    )


async def _false():
    return False


@pytest.mark.asyncio
async def test_repository_closes_owned_event_poll_session(monkeypatch):
    from app.agent_workflows import repository as repository_module

    class Session:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            await self.close()

    session = Session()
    async def list_events(_session, _run_id):
        return []

    monkeypatch.setattr(repository_module, "run_store_list_run_events", list_events)

    monkeypatch.setattr(repository_module, "async_session_maker", lambda: session)
    repo = repository_module.AgentWorkflowRepository()

    assert await repo.list_run_events("run-1") == []
    assert session.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("query failed"), asyncio.CancelledError()])
async def test_repository_closes_owned_session_when_operation_aborts(monkeypatch, failure):
    from app.agent_workflows import repository as repository_module

    class Session:
        closed = False

        async def close(self):
            self.closed = True

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            await self.close()

    session = Session()

    async def list_events(_session, _run_id):
        raise failure

    monkeypatch.setattr(repository_module, "run_store_list_run_events", list_events)
    monkeypatch.setattr(repository_module, "async_session_maker", lambda: session)
    with pytest.raises(type(failure)):
        await repository_module.AgentWorkflowRepository().list_run_events("run-1")
    assert session.closed is True


@pytest.mark.asyncio
async def test_product_run_event_stream_stops_before_poll_after_disconnect(monkeypatch):
    run = SimpleNamespace(id="run-1", thread_id="thread-1")

    class Repository:
        async def list_run_events(self, _run_id):
            raise AssertionError("disconnected stream must not poll")

    async def owned_run(_run_id, _thread_id):
        return run

    monkeypatch.setattr(api, "_owned_run_for_operation", owned_run)
    monkeypatch.setattr(api, "AgentWorkflowRepository", Repository)
    request = SimpleNamespace(is_disconnected=lambda: _true())
    response = await api.stream_agent_run_events("run-1", request=request, thread_id="thread-1")
    assert [chunk async for chunk in response.body_iterator] == []


async def _true():
    return True
