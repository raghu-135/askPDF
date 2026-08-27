from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest, AgentRuntimeResult, RuntimeTaskContext
from app.runtime.http_adapter import HttpLangGraphRuntimeAdapter, context_to_dict
from app.runtime.errors import RuntimeError
from app.runtime.catalog import result_to_product_payload


def _request() -> AgentRuntimeRequest:
    return AgentRuntimeRequest(
        run_id="run-1",
        thread_id="thread-1",
        definition_id="router",
        framework="langgraph",
        builder_id="langgraph_graph",
        input={"question": "hello"},
        trace_id="trace-1",
    )


def test_runtime_error_is_raiseable_and_keeps_wire_shape():
    error = RuntimeError("runtime_timeout", "Agent runtime timed out", retryable=True)
    try:
        raise error
    except RuntimeError as caught:
        assert caught.code == "runtime_timeout"
        assert caught.retryable is True
        assert caught.to_dict()["safe_message"] == "Agent runtime timed out"


def test_typed_projection_keeps_absent_interaction_fields_null():
    projected = result_to_product_payload(AgentRuntimeResult(status="completed", output="answer"))
    assert projected["clarification_options"] is None
    assert projected["pending_interrupt"] is None


def test_http_context_preserves_task_request_fields_as_json():
    from types import SimpleNamespace

    payload = context_to_dict(
        RuntimeExecutionContext(
            request=SimpleNamespace(
                objective="find the authors",
                task_limits={"max_sources": 3},
            ),
            task_context=RuntimeTaskContext(
                task_id="task-1",
                objective="find the authors",
                limits={"max_sources": 3},
                permissions={"use_web_search": True},
                metadata={"llm_model": "test-model", "context_window": 8192},
            ),
        )
    )

    assert payload["request_payload"] == {
        "objective": "find the authors",
        "task_limits": {"max_sources": 3},
        "runtime_execution_mode": True,
    }
    assert payload["task_context"] == {
        "task_id": "task-1",
        "objective": "find the authors",
        "todos": [],
        "artifact_manifests": [],
        "artifact_contents": {},
        "limits": {"max_sources": 3},
        "permissions": {"use_web_search": True},
        "metadata": {"llm_model": "test-model", "context_window": 8192},
        "context_data": {},
    }


@pytest.mark.asyncio
async def test_runtime_cancellation_probe_is_awaitable():
    import asyncio

    event = asyncio.Event()

    async def cancellation_probe() -> bool:
        return event.is_set()

    assert await cancellation_probe() is False
    event.set()
    assert await cancellation_probe() is True


@pytest.mark.asyncio
async def test_http_adapter_round_trips_capabilities_and_validation():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/capabilities":
            assert request.method == "POST"
            assert json.loads(request.content)["definition"]["definition_id"] == "router"
            return httpx.Response(200, json={"result": {"capabilities": {"operations": {
                "run.resume": {"support": "conditional", "owner": "runtime", "enabled": True, "semantics": "resume_from_interrupt"},
            }}}})
        return httpx.Response(200, json={"result": {"validation": {"valid": True, "issues": []}}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    capabilities = await adapter.capabilities(AgentDefinition("router", "langgraph", "langgraph_graph"))
    validation = await adapter.validate(AgentDefinition("router", "langgraph", "langgraph_graph"), {"nodes": []})
    assert "run.events" not in capabilities.operations
    assert capabilities.operations["run.resume"].support.value == "conditional"
    assert validation.valid
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_rejects_bare_success_responses_at_the_wire_boundary():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"capabilities": {"operations": {}}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    with pytest.raises(RuntimeError, match="response envelope") as caught:
        await adapter.deployment_capabilities()
    assert caught.value.code == "runtime_protocol_error"
    assert calls == ["/v1/capabilities"]
    await client.aclose()


def test_event_parser_rejects_alias_kinds_and_bare_event_shapes():
    from app.runtime.transport import event_from_dict

    with pytest.raises(ValueError):
        event_from_dict({"event_id": "evt", "run_id": "run", "sequence": 1, "kind": "node.started"})


def test_capability_parser_rejects_flat_or_malformed_payloads():
    from app.runtime.transport import capabilities_from_dict

    import pytest

    with pytest.raises(ValueError):
        capabilities_from_dict({"streaming": True})
    with pytest.raises(ValueError):
        capabilities_from_dict({"operations": {"run.start": {"support": "unknown", "enabled": True}}})
    with pytest.raises(ValueError):
        capabilities_from_dict({"operations": {"run.start": {"support": "native", "owner": "runtime", "enabled": False}}})
    with pytest.raises(ValueError):
        capabilities_from_dict({"operations": {"run.start": {"support": "native", "enabled": True}}})
    with pytest.raises(ValueError):
        capabilities_from_dict({"operations": {"run.start": {"support": "native", "owner": "adapter", "enabled": True}}})
    with pytest.raises(ValueError):
        capabilities_from_dict({"operations": {"run.future": {"support": "native", "owner": "runtime", "enabled": True}}})


@pytest.mark.asyncio
async def test_http_adapter_preserves_nonterminal_identity_and_keeps_transport_terminal_internal():
    request = _request()
    event = {"event_id": "evt-1", "run_id": request.run_id, "sequence": 1, "kind": "operation.started", "payload": {"node": "router"}}
    terminal = {"event_id": "evt-terminal", "run_id": request.run_id, "sequence": 2, "kind": "run.completed", "payload": {}, "terminal": True}

    def handler(http_request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(http_request.content)
        definition = request_payload["definition"]
        assert definition["definition_id"] == "router"
        assert definition["framework"] == "langgraph"
        assert definition["builder_id"] == "langgraph_graph"
        assert definition["capabilities"] == {}
        def body():
            yield f"id: evt-1\nevent: operation.started\ndata: {json.dumps({'event': event})}\n\n".encode()
            yield f"id: evt-terminal\nevent: run.completed\ndata: {json.dumps({'event': terminal, 'result': {'status': 'completed', 'output': 'ok'}})}\n\n".encode()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"".join(body()))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    received = []

    class Sink:
        async def emit_runtime_event(self, value):
            received.append(value)

    result = await adapter.start(request, context=RuntimeExecutionContext(resolved_spec={"nodes": []}), event_sink=Sink())
    assert result.status == "completed"
    assert result.output == "ok"
    assert [item.event_id for item in received] == ["evt-1"]
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_coalesces_output_deltas_before_product_delivery():
    request = _request()
    deltas = [
        {"event_id": f"delta-{index}", "run_id": request.run_id, "sequence": index, "kind": "output.delta", "payload": {"delta": value}}
        for index, value in enumerate(("one ", "two ", "three"), start=1)
    ]
    terminal = {"event_id": "terminal", "run_id": request.run_id, "sequence": 4, "kind": "run.completed", "payload": {}, "terminal": True}

    def handler(_request):
        frames = [f"id: {event['event_id']}\nevent: {event['kind']}\ndata: {json.dumps({'event': event})}\n\n" for event in deltas]
        frames.append(f"id: terminal\nevent: run.completed\ndata: {json.dumps({'event': terminal, 'result': {'status': 'completed', 'output': 'one two three'}})}\n\n")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content="".join(frames).encode())

    received = []

    class Sink:
        async def emit_runtime_event(self, event):
            received.append(event)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    await adapter.start(request, context=RuntimeExecutionContext(), event_sink=Sink())
    assert len(received) == 1
    assert received[0].payload == {"delta": "one two three", "chunk_count": 3}
    assert received[0].source_metadata["first_source_sequence"] == 1
    assert received[0].source_metadata["last_source_sequence"] == 3
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_flushes_output_delta_on_time_while_stream_is_open(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_OUTPUT_DELTA_FLUSH_SECONDS", "0.05")
    request = _request()
    flushed = asyncio.Event()
    delta = {"event_id": "delta-1", "run_id": request.run_id, "sequence": 1, "kind": "output.delta", "payload": {"delta": "partial"}}
    terminal = {"event_id": "terminal", "run_id": request.run_id, "sequence": 2, "kind": "run.completed", "payload": {}, "terminal": True}

    class DelayedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield f"id: delta-1\nevent: output.delta\ndata: {json.dumps({'event': delta})}\n\n".encode()
            await asyncio.wait_for(flushed.wait(), timeout=0.5)
            yield f"id: terminal\nevent: run.completed\ndata: {json.dumps({'event': terminal, 'result': {'status': 'completed', 'output': 'partial'}})}\n\n".encode()

    def handler(_request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=DelayedStream())

    class Sink:
        async def emit_runtime_event(self, event):
            assert event.kind == "output.delta"
            flushed.set()

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    result = await adapter.start(request, context=RuntimeExecutionContext(), event_sink=Sink())
    assert result.status == "completed"
    assert flushed.is_set()
    await client.aclose()


@pytest.mark.asyncio
async def test_recovery_deadline_starts_after_long_initial_stream(monkeypatch):
    request = _request()
    progress = {"event_id": "progress", "run_id": request.run_id, "sequence": 1, "kind": "output.delta", "payload": {"delta": "partial "}}
    terminal = {"event_id": "terminal", "run_id": request.run_id, "sequence": 2, "kind": "run.completed", "payload": {}, "terminal": True}
    calls = []

    def handler(http_request):
        calls.append(http_request.method)
        if http_request.method == "POST":
            time.sleep(1.05)
            body = f"id: progress\nevent: output.delta\ndata: {json.dumps({'event': progress})}\n\n"
        else:
            body = f"id: terminal\nevent: run.completed\ndata: {json.dumps({'event': terminal, 'result': {'status': 'completed', 'output': 'recovered'}})}\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    monkeypatch.setenv("AGENT_RUNTIME_RECONNECT_DEADLINE_SECONDS", "1")
    monkeypatch.setenv("AGENT_RUNTIME_RECONNECT_BACKOFF_SECONDS", "0.001")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    result = await adapter.start(request, context=RuntimeExecutionContext())
    assert result.output == "recovered"
    assert calls == ["POST", "GET"]
    await client.aclose()


@pytest.mark.asyncio
async def test_healthy_replay_may_run_longer_than_recovery_deadline(monkeypatch):
    request = _request()
    terminal = {"event_id": "terminal", "run_id": request.run_id, "sequence": 2, "kind": "run.completed", "payload": {}, "terminal": True}
    calls: list[str] = []

    class SlowReplay(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(0.06)
            yield f"id: terminal\nevent: run.completed\ndata: {json.dumps({'event': terminal, 'result': {'status': 'completed', 'output': 'recovered'}})}\n\n".encode()

    def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request.method)
        if http_request.method == "POST":
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"")
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SlowReplay())

    monkeypatch.setenv("AGENT_RUNTIME_RECONNECT_DEADLINE_SECONDS", "0.05")
    monkeypatch.setenv("AGENT_RUNTIME_RECONNECT_BACKOFF_SECONDS", "0.001")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    result = await adapter.start(request, context=RuntimeExecutionContext())
    assert result.output == "recovered"
    assert calls == ["POST", "GET"]
    await client.aclose()


@pytest.mark.asyncio
async def test_identical_runtime_binding_is_persisted_once():
    request = _request()
    binding = {"binding_type": "session", "payload": {"session_id": "upstream-1"}}
    events = [
        {"event_id": "delta-1", "run_id": request.run_id, "sequence": 1, "kind": "output.delta", "payload": {"delta": "one"}, "continuation": binding},
        {"event_id": "delta-2", "run_id": request.run_id, "sequence": 2, "kind": "output.delta", "payload": {"delta": " two"}, "continuation": binding},
        {"event_id": "terminal", "run_id": request.run_id, "sequence": 3, "kind": "run.completed", "payload": {}, "terminal": True, "continuation": binding},
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        body = "".join(
            f"id: {event['event_id']}\nevent: {event['kind']}\ndata: {json.dumps({'event': event, **({'result': {'status': 'completed', 'output': 'one two'}} if event.get('terminal') else {})})}\n\n"
            for event in events
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    class Sink:
        def __init__(self):
            self.bindings = []

        async def emit_runtime_event(self, _event):
            return None

        async def persist_runtime_binding(self, _run_id, continuation):
            self.bindings.append(continuation)

    sink = Sink()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    await adapter.start(request, context=RuntimeExecutionContext(), event_sink=sink)
    assert len(sink.bindings) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_preserves_dependency_admission_error():
    request = _request()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={
            "status": "failed",
            "error": {
                "code": "runtime_dependency_unavailable",
                "safe_message": "A dependency required by this agent is unavailable",
                "retryable": True,
                "details": {"dependency": "mcp", "missing_capability_ids": ["document_evidence"]},
            },
            "runtime_metadata": {"framework": "langgraph"},
        })

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    with pytest.raises(RuntimeError) as caught:
        await adapter.start(request, context=RuntimeExecutionContext())
    assert caught.value.code == "runtime_dependency_unavailable"
    assert caught.value.retryable is True
    assert caught.value.details["dependency"] == "mcp"
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload", "expected_code", "expected_retryable"),
    [
        (400, {"error": {"code": "invalid_request", "safe_message": "bad request", "retryable": False}}, "invalid_request", False),
        (409, {"detail": {"code": "runtime_operation_conflict", "safe_message": "terminal execution is immutable; use retry", "retryable": False}}, "runtime_operation_conflict", False),
        (503, {"error": {"code": "runtime_dependency_unavailable", "safe_message": "dependency unavailable", "retryable": True}}, "runtime_dependency_unavailable", True),
    ],
)
async def test_http_adapter_preserves_structured_json_http_errors(status, payload, expected_code, expected_retryable):
    request = _request()

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    with pytest.raises(RuntimeError) as caught:
        await adapter.cancel(request)
    assert caught.value.code == expected_code
    assert caught.value.retryable is expected_retryable
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "payload", "expected_code", "expected_retryable"),
    [
        (409, {"detail": {"code": "runtime_operation_conflict", "safe_message": "terminal execution is immutable; use retry", "retryable": False}}, "runtime_operation_conflict", False),
        (503, {"error": {"code": "runtime_dependency_unavailable", "safe_message": "dependency unavailable", "retryable": True}}, "runtime_dependency_unavailable", True),
    ],
)
async def test_http_adapter_preserves_structured_stream_http_errors(status, payload, expected_code, expected_retryable):
    request = _request()

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    with pytest.raises(RuntimeError) as caught:
        await adapter.start(request, context=RuntimeExecutionContext())
    assert caught.value.code == expected_code
    assert caught.value.retryable is expected_retryable
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_maps_network_failure_to_retryable_transport_error():
    request = _request()

    def handler(http_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=http_request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    with pytest.raises(RuntimeError) as caught:
        await adapter.cancel(request)
    assert caught.value.code == "runtime_transport_error"
    assert caught.value.retryable is True
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_maps_malformed_http_error_to_retryable_transport_error():
    request = _request()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, content=b"not-json")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    with pytest.raises(RuntimeError) as caught:
        await adapter.cancel(request)
    assert caught.value.code == "runtime_transport_error"
    assert caught.value.retryable is True
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_does_not_retry_structured_stream_conflict():
    request = _request()
    calls: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request.method + " " + http_request.url.path)
        return httpx.Response(409, json={"detail": {"code": "runtime_operation_conflict", "safe_message": "conflict", "retryable": False}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    with pytest.raises(RuntimeError) as caught:
        await adapter.start(request, context=RuntimeExecutionContext())
    assert caught.value.code == "runtime_operation_conflict"
    assert calls == ["POST /v1/runs/start"]
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_maps_no_continuation_to_none():
    request = _request()

    def handler(http_request: httpx.Request) -> httpx.Response:
        event = {
            "event_id": "terminal",
            "run_id": request.run_id,
            "sequence": 1,
            "kind": "run.completed",
            "payload": {"status": "no_continuation"},
            "terminal": True,
        }
        body = f"id: terminal\nevent: run.completed\ndata: {json.dumps({'event': event, 'result': {'status': 'no_continuation'}})}\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    assert await adapter.continue_run(request, context=RuntimeExecutionContext()) is None
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_rejects_conflicting_duplicate_event_ids():
    request = _request()
    first = {"event_id": "evt-1", "run_id": request.run_id, "sequence": 1, "kind": "operation.started", "payload": {"node": "router"}}
    conflicting = {**first, "payload": {"node": "planner"}}

    def handler(_request: httpx.Request) -> httpx.Response:
        body = (
            f"id: evt-1\nevent: operation.started\ndata: {json.dumps({'event': first})}\n\n"
            f"id: evt-1\nevent: operation.started\ndata: {json.dumps({'event': conflicting})}\n\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    with pytest.raises(RuntimeError, match="conflicting duplicate event IDs"):
        await adapter.start(request, context=RuntimeExecutionContext())
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_rejects_result_on_nonterminal_event():
    request = _request()
    event = {"event_id": "evt-1", "run_id": request.run_id, "sequence": 1, "kind": "operation.started", "payload": {}}

    def handler(_request: httpx.Request) -> httpx.Response:
        body = f"id: evt-1\nevent: operation.started\ndata: {json.dumps({'event': event, 'result': {'status': 'completed'}})}\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    with pytest.raises(RuntimeError, match="nonterminal event"):
        await adapter.start(request, context=RuntimeExecutionContext())
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_replays_after_transport_disconnect_before_terminal():
    request = _request()
    calls: list[str] = []
    progress = {"event_id": "evt-1", "run_id": request.run_id, "sequence": 1, "kind": "operation.started", "payload": {}}
    terminal = {"event_id": "evt-terminal", "run_id": request.run_id, "sequence": 2, "kind": "run.completed", "payload": {}, "terminal": True}

    def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request.method + " " + http_request.url.path)
        if http_request.method == "POST":
            raise httpx.ReadError("subscriber disconnected", request=http_request)
        body = (
            f"id: evt-1\nevent: operation.started\ndata: {json.dumps({'event': progress})}\n\n"
            f"id: evt-terminal\nevent: run.completed\ndata: {json.dumps({'event': terminal, 'result': {'status': 'completed', 'output': 'recovered'}})}\n\n"
        )
        assert http_request.url.params["after_sequence"] == "0"
        assert "last-event-id" not in http_request.headers
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    result = await adapter.start(request, context=RuntimeExecutionContext())
    assert result.output == "recovered"
    assert calls == ["POST /v1/runs/start", "GET /v1/runs/run-1/events"]
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 404])
async def test_http_adapter_does_not_replay_unstructured_http_status_failures(status_code):
    request = _request()
    calls: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request.method + " " + http_request.url.path)
        return httpx.Response(status_code, request=http_request, content=b"not json")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    with pytest.raises(RuntimeError) as caught:
        await adapter.start(request, context=RuntimeExecutionContext())

    assert caught.value.code == "runtime_transport_error"
    assert caught.value.retryable is False
    assert caught.value.details["status_code"] == status_code
    assert calls == ["POST /v1/runs/start"]
    await client.aclose()
