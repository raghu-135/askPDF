"""Opt-in smoke test against the exact pinned Hermes container."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from app.runtime.adapter import RuntimeInvocationContext
from runtime_protocol.contracts import AgentRuntimeRequest
from app.runtime.hermes_adapter import HermesRuntimeAdapter


pytestmark = pytest.mark.skipif(
    os.getenv("HERMES_RUNTIME_REAL_SMOKE", "").lower() not in {"1", "true", "yes", "on"},
    reason="requires HERMES_RUNTIME_REAL_SMOKE=true and the pinned real-Hermes Compose profile",
)

_LEGACY_PROFILE_MODEL = "askpdf-runtime-selected"


class _Sink:
    def __init__(self) -> None:
        self.events = []
        self.started = asyncio.Event()
        self.continuation = None

    async def emit(self, event) -> None:
        self.events.append(event)
        if event.kind == "run.started":
            self.continuation = event.continuation
            self.started.set()


def _request(prompt: str) -> AgentRuntimeRequest:
    unique = uuid.uuid4().hex
    return AgentRuntimeRequest(
        run_id=f"real-hermes-{unique}",
        thread_id=f"real-hermes-thread-{unique}",
        definition_id="hermes_rag_agent",
        framework="hermes",
        builder_id="hermes_agent",
        input={"question": prompt},
        options={"llm_model": _LEGACY_PROFILE_MODEL, "llm_provider": "lmstudio"},
    )


def _context() -> RuntimeInvocationContext:
    return RuntimeInvocationContext(resolved_spec={"definition_version": 1, "config": {"system_prompt": "Respond concisely.", "model": _LEGACY_PROFILE_MODEL, "provider": "lmstudio", "mcp_server": "askpdf", "allowed_tool_ids": [], "max_duration_seconds": 120}})


@pytest.mark.asyncio
async def test_pinned_real_hermes_completion_stream_and_session_capture(monkeypatch) -> None:
    monkeypatch.setenv("COMPOSE_PROFILES", "hermes")
    assert int(os.environ["HERMES_MODEL_CONTEXT_LENGTH"]) >= 2048
    adapter = HermesRuntimeAdapter(base_url=os.getenv("HERMES_RUNTIME_URL", "http://localhost:8201"))
    sink = _Sink()
    try:
        result = await adapter.start(_request("Reply with exactly: smoke-ok"), context=_context(), event_sink=sink)
        assert result.status == "completed"
        assert result.continuation is not None
        assert result.continuation.payload["session_id"]
        assert any(event.kind == "output.delta" for event in sink.events)
        assert any(event.kind == "run.completed" for event in sink.events)
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_pinned_real_hermes_cancellation(monkeypatch) -> None:
    monkeypatch.setenv("COMPOSE_PROFILES", "hermes")
    assert int(os.environ["HERMES_MODEL_CONTEXT_LENGTH"]) >= 2048
    adapter = HermesRuntimeAdapter(base_url=os.getenv("HERMES_RUNTIME_URL", "http://localhost:8201"))
    request = _request("Work slowly and continue until stopped.")
    sink = _Sink()
    task = asyncio.create_task(adapter.start(request, context=_context(), event_sink=sink))
    try:
        await asyncio.wait_for(sink.started.wait(), timeout=30)
        bound_request = AgentRuntimeRequest(**{**request.__dict__, "continuation": sink.continuation})
        response = await adapter.cancel(bound_request)
        assert response["status"] == "cancelled"
        result = await asyncio.wait_for(task, timeout=30)
        assert result.status == "cancelled"
    finally:
        if not task.done():
            task.cancel()
        await adapter.aclose()
