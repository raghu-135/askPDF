from __future__ import annotations

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
    response = await api.stream_agent_run_events("run-1", thread_id="thread-1", after_sequence=0)

    chunks = [chunk async for chunk in response.body_iterator]
    payloads = [
        json.loads(line.removeprefix("data: "))
        for chunk in chunks
        for line in chunk.splitlines()
        if line.startswith("data:")
    ]

    assert len(payloads) == 1
    assert payloads[0]["terminal"] is True
