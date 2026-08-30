from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest
from httpx import ASGITransport

from app.runtime.contracts import AgentRuntimeEvent, AgentRuntimeResult
from app.runtime.contracts import ContinuationBinding
from runtime_service.api import create_app
from runtime_service.dependencies import langgraph_dependency_requirements
from runtime_service.execution_store import ExecutionStore


class _FakeAdapter:
    async def prepare_execution_context(self, context):
        return context


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


def test_langgraph_dependency_requirements_only_admit_chat_model_to_provider() -> None:
    requirements = langgraph_dependency_requirements({
        "request": {
            "options": {
                "llm_model": "chat-model",
                "embedding_model": "BAAI/bge-m3",
            },
        },
        "context": {
            "embedding_model": "BAAI/bge-m3",
            "request_payload": {
                "embedding_model": "BAAI/bge-m3",
            },
            "resolved_spec": {
                "config": {
                    "allowed_tool_ids": [],
                },
            },
        },
    })

    assert requirements["provider"] == {"chat-model"}


def test_langgraph_dependency_requirements_reads_neutral_task_model() -> None:
    requirements = langgraph_dependency_requirements({
        "request": {"options": {}},
        "context": {
            "task_context": {"metadata": {"llm_model": "task-chat-model"}},
            "resolved_spec": {"config": {"allowed_tool_ids": []}},
        },
    })

    assert requirements["provider"] == {"task-chat-model"}


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
async def test_runtime_prepares_neutral_task_context_before_langgraph_start(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = False

    class FakeAdapter(_FakeAdapter):
        async def prepare_execution_context(self, context):
            nonlocal prepared
            prepared = True
            assert context.task_context is not None
            assert context.task_context.metadata["llm_model"] == "task-model"
            assert context.task_context.permissions["use_web_search"] is True
            return replace(
                context,
                request=SimpleNamespace(
                    question=context.task_context.objective,
                    llm_model=context.task_context.metadata["llm_model"],
                ),
            )

        async def start(self, request, *, context, event_sink=None):
            assert context.request.question == "Research transport boundaries"
            assert context.request.llm_model == "task-model"
            return AgentRuntimeResult(status="completed", output={"answer": "ok"})

    monkeypatch.setattr("app.runtime.langgraph_adapter.LangGraphRuntimeAdapter", FakeAdapter)
    monkeypatch.setattr("runtime_service.api.langgraph_dependency_requirements", lambda payload: {})
    payload = _payload("run-task-context")
    payload["context"] = {
        "task_context": {
            "task_id": "task-1",
            "objective": "Research transport boundaries",
            "todos": [],
            "artifact_manifests": [],
            "artifact_contents": {},
            "limits": {"max_sources": 3},
            "permissions": {"use_web_search": True},
            "metadata": {"llm_model": "task-model", "context_window": 8192},
            "context_data": {},
        },
    }
    app = create_app(execution_store=ExecutionStore())
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        events = await _read_events(client, "POST", "/v1/runs/start", json=payload)

    assert prepared is True
    assert events[-1]["result"]["status"] == "completed"


@pytest.mark.asyncio
async def test_completed_run_event_replay_and_repeated_start_are_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    class FakeAdapter(_FakeAdapter):
        async def start(self, request, *, context, event_sink=None):
            nonlocal calls
            calls += 1
            if event_sink is not None:
                await event_sink.emit_runtime_event(
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

    class FakeAdapter(_FakeAdapter):
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

    class FakeAdapter(_FakeAdapter):
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

    class FakeAdapter(_FakeAdapter):
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


@pytest.mark.asyncio
async def test_pause_request_is_persisted_for_external_execution() -> None:
    store = ExecutionStore()
    await store.create("run-pause", "start", _request("run-pause"), _payload("run-pause"))
    app = create_app(execution_store=store)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        response = await client.post("/v1/runs/run-pause/pause", json={"request": _request("run-pause")})

    assert response.status_code == 200
    assert response.json()["result"]["status"] == "pause_requested"
    assert await store.is_pause_requested("run-pause") is True


@pytest.mark.asyncio
async def test_cancellation_checker_stops_work_and_persists_one_terminal_event(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()

    class BlockingAdapter(_FakeAdapter):
        async def start(self, request, *, context, event_sink=None):
            started.set()
            try:
                while not await context.cancellation_checker():
                    await asyncio.sleep(0.001)
                raise asyncio.CancelledError
            finally:
                stopped.set()

    monkeypatch.setattr("app.runtime.langgraph_adapter.LangGraphRuntimeAdapter", BlockingAdapter)
    store = ExecutionStore()
    app = create_app(execution_store=store)
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://runtime") as client:
        stream_task = asyncio.create_task(
            _read_events(client, "POST", "/v1/runs/start", json=_payload("run-blocking-cancel"))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        response = await client.post(
            "/v1/runs/run-blocking-cancel/cancel",
            json={"request": _request("run-blocking-cancel")},
        )
        events = await asyncio.wait_for(stream_task, timeout=3)

    assert response.status_code == 200
    assert stopped.is_set()
    terminals = [event for event in events if event["event"]["terminal"]]
    assert len(terminals) == 1
    assert terminals[0]["event"]["kind"] == "run.cancelled"
    assert terminals[0]["result"]["status"] == "cancelled"
    record = await store.get("run-blocking-cancel")
    assert record is not None and record.status == "cancelled"
