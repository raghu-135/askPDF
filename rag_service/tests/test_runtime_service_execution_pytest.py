from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from httpx import ASGITransport

from app.runtime.contracts import AgentRuntimeEvent, AgentRuntimeResult
from app.runtime.contracts import ContinuationBinding
from runtime_service.api import create_app
from runtime_service.execution_store import ExecutionStore


def _request(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "thread_id": "thread-1",
        "definition_id": "router_rag_agent",
        "framework": "langgraph",
        "builder_id": "langgraph_graph",
        "input": {"question": "hello"},
        "options": {},
    }


def _payload(run_id: str) -> dict:
    return {
        "request": _request(run_id),
        "context": {},
        "definition": {
            "definition_id": "router_rag_agent",
            "framework": "langgraph",
            "builder_id": "langgraph_graph",
            "capabilities": {},
            "definition_metadata": {},
        },
    }


async def _read_events(client: httpx.AsyncClient, method: str, url: str, **kwargs: object) -> list[dict]:
    async with client.stream(method, url, **kwargs) as response:
        assert response.status_code == 200
        body = await response.aread()
    events = []
    for block in body.decode().split("\n\n"):
        data_line = next((line for line in block.splitlines() if line.startswith("data:")), None)
        if data_line:
            events.append(json.loads(data_line[5:].strip()))
    return events


@pytest.mark.asyncio
async def test_completed_run_event_replay_and_repeated_start_are_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class FakeAdapter:
        async def start(self, request, *, context, event_sink=None):
            nonlocal calls
            calls += 1
            if event_sink is not None:
                await event_sink.emit(
                    AgentRuntimeEvent(
                        event_id=f"{request.run_id}:progress",
                        run_id=request.run_id,
                        sequence=1,
                        kind="run.progress",
                        payload={"step": 1},
                    )
                )
            return AgentRuntimeResult(status="completed", output={"answer": "ok"})

    monkeypatch.setattr("app.runtime.langgraph_adapter.LangGraphRuntimeAdapter", FakeAdapter)
    store = ExecutionStore()
    app = create_app(execution_store=store)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        first = await _read_events(client, "POST", "/v1/runs/start", json=_payload("run-complete"))
        repeated = await _read_events(client, "POST", "/v1/runs/start", json=_payload("run-complete"))

    # A new app instance represents a process restart while the durable store
    # and terminal event remain available.
    restarted_app = create_app(execution_store=store)
    restarted_transport = ASGITransport(app=restarted_app)
    async with httpx.AsyncClient(transport=restarted_transport, base_url="http://runtime") as client:
        replay = await _read_events(client, "GET", "/v1/runs/run-complete/events")

    assert calls == 1
    assert first[-1]["event"]["terminal"] is True
    assert replay[-1]["event"]["terminal"] is True
    assert repeated[-1]["event"]["terminal"] is True


@pytest.mark.asyncio
async def test_two_simultaneous_subscribers_start_one_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    started = asyncio.Event()

    class FakeAdapter:
        async def start(self, request, *, context, event_sink=None):
            nonlocal calls
            calls += 1
            started.set()
            await asyncio.sleep(0.03)
            return AgentRuntimeResult(status="completed", output={"answer": "ok"})

    monkeypatch.setattr("app.runtime.langgraph_adapter.LangGraphRuntimeAdapter", FakeAdapter)
    app = create_app()
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        results = await asyncio.gather(
            _read_events(client, "POST", "/v1/runs/start", json=_payload("run-shared")),
            _read_events(client, "POST", "/v1/runs/start", json=_payload("run-shared")),
        )

    assert started.is_set()
    assert calls == 1
    assert all(events[-1]["event"]["terminal"] for events in results)


@pytest.mark.asyncio
async def test_resume_after_a_terminal_start_requires_explicit_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeAdapter:
        async def start(self, request, *, context, event_sink=None):
            calls.append("start")
            return AgentRuntimeResult(
                status="completed",
                output={"answer": "paused result"},
                continuation=ContinuationBinding("checkpoint", {"id": "cp-1"}),
            )

        async def resume(self, request, *, interrupt, context, event_sink=None):
            calls.append("resume")
            return AgentRuntimeResult(status="completed", output={"answer": "resumed result"})

    monkeypatch.setattr("app.runtime.langgraph_adapter.LangGraphRuntimeAdapter", FakeAdapter)
    app = create_app(execution_store=ExecutionStore())
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        first = await _read_events(client, "POST", "/v1/runs/start", json=_payload("run-hitl"))
        response = await client.post(
            "/v1/runs/run-hitl/resume",
            json={**_payload("run-hitl"), "interrupt": {"decision": "approve"}},
        )

    assert calls == ["start"]
    assert first[-1]["result"]["output"]["answer"] == "paused result"
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_explicit_retry_creates_one_new_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class FakeAdapter:
        async def start(self, request, *, context, event_sink=None):
            nonlocal calls
            calls += 1
            return AgentRuntimeResult(status="completed", output={"answer": f"attempt-{calls}"})

    monkeypatch.setattr("app.runtime.langgraph_adapter.LangGraphRuntimeAdapter", FakeAdapter)
    app = create_app(execution_store=ExecutionStore())
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        first = await _read_events(client, "POST", "/v1/runs/start", json=_payload("run-explicit-retry"))
        retry_payload = {
            "attempt_id": "retry-operation-1",
            "source_attempt": 1,
            "operation": "start",
            "request": _request("run-explicit-retry"),
            "definition": _payload("run-explicit-retry")["definition"],
        }
        retried = await _read_events(client, "POST", "/v1/runs/run-explicit-retry/retry", json=retry_payload)
        repeated = await _read_events(client, "POST", "/v1/runs/run-explicit-retry/retry", json=retry_payload)
        retry_two_payload = {
            "attempt_id": "retry-operation-2",
            "source_attempt": 2,
            "operation": "start",
            "request": _request("run-explicit-retry"),
            "definition": _payload("run-explicit-retry")["definition"],
        }
        second_retry = await _read_events(client, "POST", "/v1/runs/run-explicit-retry/retry", json=retry_two_payload)
        delayed_repeated = await _read_events(client, "POST", "/v1/runs/run-explicit-retry/retry", json=retry_payload)

    assert calls == 3
    assert first[-1]["result"]["output"]["answer"] == "attempt-1"
    assert retried[-1]["result"]["output"]["answer"] == "attempt-2"
    assert repeated[-1]["result"]["output"]["answer"] == "attempt-2"
    assert second_retry[-1]["result"]["output"]["answer"] == "attempt-3"
    assert delayed_repeated[-1]["result"]["output"]["answer"] == "attempt-2"


@pytest.mark.asyncio
async def test_cancel_unknown_run_returns_404_without_creating_state() -> None:
    store = ExecutionStore()
    app = create_app(execution_store=store)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        response = await client.post("/v1/runs/missing/cancel", json={"request": _request("missing")})

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "runtime_run_not_found"
    assert await store.get("missing") is None


@pytest.mark.asyncio
async def test_cancel_active_and_terminal_runs_are_idempotent() -> None:
    store = ExecutionStore()
    await store.create("run-cancel", "start", _request("run-cancel"), _payload("run-cancel"))
    app = create_app(execution_store=store)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        first = await client.post("/v1/runs/run-cancel/cancel", json={"request": _request("run-cancel")})
        repeated = await client.post("/v1/runs/run-cancel/cancel", json={"request": _request("run-cancel")})
        await store.set_status("run-cancel", "cancelled")
        terminal = await client.post("/v1/runs/run-cancel/cancel", json={"request": _request("run-cancel")})

    assert first.status_code == 200
    assert first.json()["result"]["status"] == "cancellation_requested"
    assert repeated.json()["result"]["status"] == "cancellation_requested"
    assert terminal.status_code == 200
    assert terminal.json()["result"] == {
        "run_id": "run-cancel",
        "status": "cancelled",
        "cancellation_requested": False,
        "no_op": True,
    }
