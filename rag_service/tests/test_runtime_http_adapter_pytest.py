from __future__ import annotations

import json

import httpx
import pytest

from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest, AgentRuntimeResult
from app.runtime.http_adapter import HttpLangGraphRuntimeAdapter, context_to_dict
from app.runtime.errors import RuntimeError
from app.runtime.langgraph_compat import legacy_result_from_runtime


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


def test_legacy_projection_keeps_absent_interaction_fields_null():
    projected = legacy_result_from_runtime(AgentRuntimeResult(status="completed", output="answer"))
    assert projected["clarification_options"] is None
    assert projected["pending_interrupt"] is None


def test_http_context_preserves_task_request_fields_as_json():
    from types import SimpleNamespace

    payload = context_to_dict(
        RuntimeExecutionContext(
            request=SimpleNamespace(
                objective="find the authors",
                task_limits={"max_sources": 3},
            )
        )
    )

    assert payload["request_payload"] == {
        "objective": "find the authors",
        "task_limits": {"max_sources": 3},
        "runtime_execution_mode": True,
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
            return httpx.Response(200, json={"contract_version": 1, "capabilities": {"operations": {
                "run.events": {"support": "native", "enabled": True},
                "run.resume": {"support": "conditional", "enabled": True, "semantics": "resume_from_interrupt"},
            }}})
        return httpx.Response(200, json={"contract_version": 1, "validation": {"valid": True, "issues": []}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    capabilities = await adapter.capabilities(AgentDefinition("router", "langgraph", "langgraph_graph"))
    validation = await adapter.validate(AgentDefinition("router", "langgraph", "langgraph_graph"), {"nodes": []})
    assert capabilities.operations["run.events"].enabled
    assert capabilities.operations["run.resume"].support.value == "conditional"
    assert validation.valid
    await client.aclose()


def test_capability_parser_rejects_flat_or_malformed_payloads():
    from app.runtime.transport import capabilities_from_dict

    import pytest

    with pytest.raises(ValueError):
        capabilities_from_dict({"streaming": True})
    with pytest.raises(ValueError):
        capabilities_from_dict({"operations": {"run.start": {"support": "unknown", "enabled": True}}})
    with pytest.raises(ValueError):
        capabilities_from_dict({"operations": {"run.start": {"support": "native", "enabled": False}}})


@pytest.mark.asyncio
async def test_http_adapter_preserves_event_identity_and_terminal_result():
    request = _request()
    event = {"event_id": "evt-1", "run_id": request.run_id, "sequence": 1, "kind": "node.started", "payload": {"node": "router"}}
    terminal = {"event_id": "evt-terminal", "run_id": request.run_id, "sequence": 2, "kind": "run.completed", "payload": {}, "terminal": True}

    def handler(http_request: httpx.Request) -> httpx.Response:
        def body():
            yield f"id: evt-1\nevent: node.started\ndata: {json.dumps({'event': event})}\n\n".encode()
            yield f"id: evt-terminal\nevent: run.completed\ndata: {json.dumps({'event': terminal, 'result': {'status': 'completed', 'output': 'ok'}})}\n\n".encode()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=b"".join(body()))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    received = []

    class Sink:
        async def emit(self, value):
            received.append(value)

    result = await adapter.start(request, context=RuntimeExecutionContext(resolved_spec={"nodes": []}), event_sink=Sink())
    assert result.status == "completed"
    assert result.output == "ok"
    assert [item.event_id for item in received] == ["evt-1", "evt-terminal"]
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_preserves_dependency_admission_error():
    request = _request()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={
            "contract_version": 1,
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
        (400, {"contract_version": 1, "error": {"code": "invalid_request", "safe_message": "bad request", "retryable": False}}, "invalid_request", False),
        (409, {"detail": {"code": "runtime_operation_conflict", "safe_message": "terminal execution is immutable; use retry", "retryable": False}}, "runtime_operation_conflict", False),
        (503, {"contract_version": 1, "error": {"code": "runtime_dependency_unavailable", "safe_message": "dependency unavailable", "retryable": True}}, "runtime_dependency_unavailable", True),
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
async def test_http_adapter_translates_events_for_legacy_two_argument_sink():
    request = _request()
    event = {"event_id": "evt-1", "run_id": request.run_id, "sequence": 1, "kind": "node.started", "payload": {"node": "router"}}

    def handler(http_request: httpx.Request) -> httpx.Response:
        body = (
            f"id: evt-1\nevent: node.started\ndata: {json.dumps({'event': event})}\n\n"
            f"id: evt-terminal\nevent: run.completed\ndata: {json.dumps({'event': {**event, 'event_id': 'evt-terminal', 'sequence': 2, 'kind': 'run.completed', 'terminal': True}, 'result': {'status': 'completed'}})}\n\n"
        )
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    received = []

    class LegacySink:
        async def emit(self, event_name, data):
            received.append((event_name, data))

    await adapter.start(request, context=RuntimeExecutionContext(), event_sink=LegacySink())
    assert received[0] == ("node.started", {"node": "router"})
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_maps_no_continuation_to_none():
    request = _request()

    def handler(http_request: httpx.Request) -> httpx.Response:
        event = {
            "event_id": "terminal",
            "run_id": request.run_id,
            "sequence": 1,
            "kind": "run.continuation_empty",
            "payload": {"status": "no_continuation"},
            "terminal": True,
        }
        body = f"id: terminal\nevent: run.continuation_empty\ndata: {json.dumps({'event': event, 'result': {'status': 'no_continuation'}})}\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HttpLangGraphRuntimeAdapter("http://runtime", client=client)
    assert await adapter.continue_run(request, context=RuntimeExecutionContext()) is None
    await client.aclose()


@pytest.mark.asyncio
async def test_http_adapter_rejects_conflicting_duplicate_event_ids():
    request = _request()
    first = {"event_id": "evt-1", "run_id": request.run_id, "sequence": 1, "kind": "node.started", "payload": {"node": "router"}}
    conflicting = {**first, "payload": {"node": "planner"}}

    def handler(_request: httpx.Request) -> httpx.Response:
        body = (
            f"id: evt-1\nevent: node.started\ndata: {json.dumps({'event': first})}\n\n"
            f"id: evt-1\nevent: node.started\ndata: {json.dumps({'event': conflicting})}\n\n"
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
    event = {"event_id": "evt-1", "run_id": request.run_id, "sequence": 1, "kind": "node.started", "payload": {}}

    def handler(_request: httpx.Request) -> httpx.Response:
        body = f"id: evt-1\nevent: node.started\ndata: {json.dumps({'event': event, 'result': {'status': 'completed'}})}\n\n"
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
    progress = {"event_id": "evt-1", "run_id": request.run_id, "sequence": 1, "kind": "node.started", "payload": {}}
    terminal = {"event_id": "evt-terminal", "run_id": request.run_id, "sequence": 2, "kind": "run.completed", "payload": {}, "terminal": True}

    def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request.method + " " + http_request.url.path)
        if http_request.method == "POST":
            raise httpx.ReadError("subscriber disconnected", request=http_request)
        body = (
            f"id: evt-1\nevent: node.started\ndata: {json.dumps({'event': progress})}\n\n"
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
