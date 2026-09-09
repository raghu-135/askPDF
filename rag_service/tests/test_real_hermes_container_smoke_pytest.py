"""Opt-in smoke test against the exact pinned Hermes container."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from app.agent_workflows.builtin_workflows import load_builtin_workflows
from app.mcp.execution_context_token import issue_execution_context_token
from app.runtime.adapter import RuntimeInvocationContext
from runtime_protocol.contracts import AgentDefinition, AgentRuntimeRequest
from app.runtime.hermes_builder import HermesBuilderProvider
from app.runtime.hermes_adapter import HermesRuntimeAdapter
from app.tools.context import ToolInvocationContext


pytestmark = pytest.mark.skipif(
    os.getenv("HERMES_RUNTIME_REAL_SMOKE", "").lower() not in {"1", "true", "yes", "on"},
    reason="requires HERMES_RUNTIME_REAL_SMOKE=true and the pinned real-Hermes Compose profile",
)

_TEST_MODEL = os.getenv("HERMES_MODEL", "hermes-runtime-deterministic-hermes")


class _Sink:
    def __init__(self) -> None:
        self.events = []
        self.started = asyncio.Event()
        self.continuation = None

    async def emit_runtime_event(self, event) -> None:
        self.events.append(event)
        if event.kind == "run.started":
            self.continuation = event.continuation
            self.started.set()


async def _invocation(prompt: str) -> tuple[AgentRuntimeRequest, RuntimeInvocationContext]:
    unique = uuid.uuid4().hex
    run_id = f"real-hermes-{unique}"
    thread_id = f"real-hermes-thread-{unique}"
    definition = AgentDefinition(
        definition_id="hermes_rag_agent",
        framework="hermes",
        builder_id="hermes_agent",
        category="deep",
    )
    builtin = next(
        item for item in load_builtin_workflows()
        if item["builtin_key"] == "hermes_rag_agent"
    )
    spec = dict(builtin["spec_json"])
    context_window = int(os.environ["HERMES_MODEL_CONTEXT_LENGTH"])
    allowed_tools = list(spec["config"]["allowed_tool_ids"])
    token = issue_execution_context_token(
        ToolInvocationContext(
            thread_id=thread_id,
            run_id=run_id,
            embedding_model="hermes-runtime-deterministic-embedding",
            context_window=context_window,
            extensions={"task_id": run_id, "llm_model": _TEST_MODEL},
        ),
        task_id=run_id,
        allowed_tools=allowed_tools,
    )
    request = AgentRuntimeRequest(
        run_id=f"real-hermes-{unique}",
        thread_id=f"real-hermes-thread-{unique}",
        task_id=run_id,
        definition_id="hermes_rag_agent",
        framework="hermes",
        builder_id="hermes_agent",
        input={"question": prompt, "mcp_execution_context_token": token},
        options={
            "llm_model": _TEST_MODEL,
            "llm_provider": "lmstudio",
            "context_window": context_window,
        },
    )
    resolved = await HermesBuilderProvider().resolve(
        definition,
        spec,
        request_overrides={
            "llm_model": _TEST_MODEL,
            "context_window": context_window,
        },
    )
    return request, RuntimeInvocationContext(resolved_spec=resolved)


@pytest.mark.asyncio
async def test_pinned_real_hermes_completion_stream_and_session_capture(monkeypatch) -> None:
    monkeypatch.setenv("COMPOSE_PROFILES", "hermes")
    assert int(os.environ["HERMES_MODEL_CONTEXT_LENGTH"]) >= 2048
    adapter = HermesRuntimeAdapter(base_url=os.getenv("HERMES_RUNTIME_URL", "http://localhost:8201"))
    sink = _Sink()
    request, context = await _invocation("Reply with exactly: smoke-ok")
    try:
        result = await adapter.start(request, context=context, event_sink=sink)
        assert result.status == "completed"
        assert result.continuation is not None
        assert result.continuation.payload["session_id"]
        assert any(event.kind == "output.delta" for event in sink.events)
        # Terminal events are consumed by the connector to build the returned
        # result; only non-terminal trace events are forwarded to product sinks.
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_pinned_real_hermes_cancellation(monkeypatch) -> None:
    monkeypatch.setenv("COMPOSE_PROFILES", "hermes")
    assert int(os.environ["HERMES_MODEL_CONTEXT_LENGTH"]) >= 2048
    adapter = HermesRuntimeAdapter(base_url=os.getenv("HERMES_RUNTIME_URL", "http://localhost:8201"))
    request, context = await _invocation("Work slowly and continue until stopped.")
    sink = _Sink()
    task = asyncio.create_task(adapter.start(request, context=context, event_sink=sink))
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
