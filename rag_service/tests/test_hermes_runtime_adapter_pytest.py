import builtins
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock
import httpx
import pytest
from fastapi.testclient import TestClient

from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest, ContinuationBinding, RuntimeApprovalResponse, RuntimeOperationId, RuntimeSteeringInput
from app.runtime.hermes_adapter import HermesRuntimeAdapter
from app.runtime.capability_resolver import capabilities_for_definition, discover_adapter_capabilities
from app.runtime.errors import RuntimeError
from app.runtime.registry import RuntimeRegistry
from hermes_runtime import api as hermes_api
from hermes_runtime.compatibility import HERMES_REVISION
from hermes_runtime.execution_store import HermesExecutionStore


@pytest.mark.asyncio
async def test_hermes_adapter_has_independent_identity():
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    assert adapter.framework == "hermes"
    assert adapter.builder_id == "hermes_agent"


@pytest.mark.asyncio
async def test_hermes_definition_capabilities_apply_task_policy(monkeypatch):
    monkeypatch.setenv("COMPOSE_PROFILES", "hermes")
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "32768")
    monkeypatch.setenv("HERMES_MODEL_PROVIDER", "lmstudio")
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    adapter._json = AsyncMock(return_value={
        "capabilities": {
            "operations": {
                "run.start": {"support": "native", "owner": "runtime", "enabled": True},
            }
        }
    })

    deployment = await adapter.deployment_capabilities()
    agent_definition = AgentDefinition("hermes_rag_agent", "hermes", "hermes_agent")
    registry = RuntimeRegistry(adapters=[adapter])
    definition = await capabilities_for_definition(agent_definition, registry=registry)

    assert "task.pause" not in deployment.operations
    assert definition.operations["task.pause"].enabled is False
    assert definition.operations["task.pause"].disabled_reason == "definition_not_task_runtime"


@pytest.mark.asyncio
async def test_hermes_malformed_capabilities_are_structured(monkeypatch):
    monkeypatch.setenv("COMPOSE_PROFILES", "hermes")
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "32768")
    monkeypatch.setenv("HERMES_MODEL_PROVIDER", "lmstudio")
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    adapter._json = AsyncMock(return_value={"capabilities": {"operations": {"run.start": {"enabled": True}}}})

    with pytest.raises(RuntimeError) as caught:
        await adapter.deployment_capabilities()

    assert caught.value.code == "runtime_protocol_error"


@pytest.mark.asyncio
async def test_hermes_capability_discovery_fails_closed_while_disabled(monkeypatch):
    monkeypatch.delenv("COMPOSE_PROFILES", raising=False)
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    adapter._json = AsyncMock()
    definition = AgentDefinition("hermes_rag_agent", "hermes", "hermes_agent")

    capabilities, error = await discover_adapter_capabilities(adapter)

    assert capabilities is not None
    assert error["code"] == "runtime_disabled"
    assert capabilities.operations[RuntimeOperationId.TASK_START].enabled is False
    assert capabilities.operations[RuntimeOperationId.TASK_START].disabled_reason == "runtime_unavailable"
    adapter._json.assert_not_awaited()


@pytest.mark.asyncio
async def test_hermes_resume_is_explicitly_unsupported():
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    request = AgentRuntimeRequest("run-1", "thread-1", "hermes_rag_agent", "hermes", "hermes_agent")
    with pytest.raises(RuntimeError) as error:
        await adapter.resume(request, interrupt={}, context=None)
    assert error.value.code == "runtime_capability_unsupported"
    assert error.value.details["operation_id"] == "run.resume"


def test_hermes_capabilities_disable_live_steering_and_expose_no_steer_route(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_API_URL", "http://hermes.test")
    monkeypatch.setenv("HERMES_RUNTIME_STATE_PATH", str(tmp_path / "state.json"))

    with TestClient(hermes_api.create_app()) as client:
        capabilities = client.get("/v1/capabilities")
        steer = client.post("/v1/runs/run-1/steer", json={})

    assert capabilities.status_code == 200
    descriptor = capabilities.json()["result"]["capabilities"]["operations"]["run.steer_live"]
    assert descriptor == {
        "support": "unsupported",
        "owner": "runtime",
        "enabled": False,
        "disabled_reason": "runtime_capability_unsupported",
    }
    assert steer.status_code == 404


def test_conflicting_start_stops_the_existing_upstream_execution(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_API_URL", "http://hermes.test")
    state_path = tmp_path / "state.json"
    monkeypatch.setenv("HERMES_RUNTIME_STATE_PATH", str(state_path))
    payload = {
        "request": {
            "run_id": "run-conflict",
            "definition_id": "hermes_rag_agent",
            "framework": "hermes",
            "builder_id": "hermes_agent",
            "input": {"question": "original"},
            "options": {},
        },
        "context": {"resolved_spec": {"config": {}}},
    }
    store = HermesExecutionStore(str(state_path))
    store.create("run-conflict", payload)
    continuation = {
        "binding_type": "hermes_session",
        "payload": {
            "session_id": "session-1",
            "upstream_run_id": "upstream-1",
            "runtime_profile": "profile-1",
        },
    }
    store.update("run-conflict", status="running", continuation=continuation)
    stop = AsyncMock(return_value={"confirmed": True, "status": "cancelled", "acknowledged_status": "stopping"})
    monkeypatch.setattr(hermes_api, "_stop_and_confirm_upstream_run", stop)

    client = TestClient(hermes_api.create_app())
    response = client.post(
        "/v1/runs/start",
        json={
            **payload,
            "request": {**payload["request"], "input": {"question": "replacement"}},
        },
    )

    assert response.status_code == 409
    stop.assert_awaited_once_with(
        "http://hermes.test",
        "profile-1",
        "upstream-1",
        {"X-Hermes-Session-Id": "session-1"},
    )
    result = HermesExecutionStore(str(state_path)).records["run-conflict"]
    assert result["status"] == "cancelled"


def test_hermes_continuation_binding_is_opaque():
    binding = ContinuationBinding("hermes_session", {"session_id": "session-1"})
    assert binding.to_dict()["payload"]["session_id"] == "session-1"


def test_document_task_context_reinforces_progressive_tool_disclosure():
    value = hermes_api._task_input_with_context(
        "Summarize the acknowledgement.",
        {
            "objective": "Summarize the acknowledgement.",
            "documents": [{"file_hash": "file-1", "name": "paper.pdf"}],
        },
    )

    assert "tool_search` searches the deferred tool catalog, not document contents" in value
    assert "semantic search uploaded document file_hash" in value
    assert "`limit`" not in value
    assert value.index("Hermes bridge requirement") < value.index("askPDF task context:")


def test_task_context_without_documents_does_not_require_document_discovery():
    value = hermes_api._task_input_with_context(
        "Research a topic.",
        {"objective": "Research a topic.", "documents": []},
    )

    assert "Hermes bridge requirement" not in value
    assert "askPDF task context:" in value


def test_initial_tool_requirement_is_enabled_only_for_document_backed_tasks():
    assert hermes_api._requires_initial_tool({"documents": [{"file_hash": "file-1"}]}) is True
    assert hermes_api._requires_initial_tool({"documents": []}) is False
    assert hermes_api._requires_initial_tool(None) is False


@pytest.mark.parametrize(
    ("upstream", "neutral"),
    [
        ("message.delta", "output.delta"),
        ("tool.started", "tool.started"),
        ("tool.completed", "tool.completed"),
        ("tool.failed", "tool.failed"),
        ("reasoning.available", "reasoning.available"),
        ("approval.request", "approval.requested"),
        ("run.completed", "run.completed"),
        ("run.failed", "run.failed"),
        ("run.cancelled", "run.cancelled"),
        ("subagent.start", "subagent.started"),
        ("subagent.complete", "subagent.completed"),
        ("future.event", "runtime.event"),
    ],
)
def test_pinned_hermes_event_mapping(upstream, neutral):
    assert hermes_api._hermes_event_kind("message", {"event": upstream}) == neutral


def test_checked_in_event_fixtures_are_data_only_and_match_the_pin():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "hermes" / "run_events.json").read_text())
    assert fixture["hermes_revision"] == HERMES_REVISION
    upstream_events = {event["event"] for values in fixture.values() if isinstance(values, list) for event in values}
    assert {"message.delta", "tool.started", "tool.completed", "reasoning.available", "approval.request", "run.completed", "run.failed", "run.cancelled", "subagent.start", "subagent.complete", "future.event"} <= upstream_events


def test_hermes_tool_events_are_normalized_and_argument_values_are_removed():
    kind, payload = hermes_api._normalized_tool_payload(
        "tool.completed",
        {
            "tool": "search_document_by_id",
            "request_id": "request-7",
            "arguments": {"query": "secret query", "file_hash": "secret hash"},
            "source_count": 55,
        },
    )

    assert kind == "tool.completed"
    assert payload["tool_name"] == "search_document_by_id"
    assert payload["tool_call_id"] == "request-7"
    assert payload["provided_argument_names"] == ["file_hash", "query"]
    assert payload["result_count"] == 55
    assert payload["ok"] is True
    assert "arguments" not in payload


def test_hermes_failed_tool_completion_is_projected_as_failure():
    kind, payload = hermes_api._normalized_tool_payload(
        "tool.completed",
        {"tool": "tool_call", "error": {"code": "invalid_arguments"}},
    )

    assert kind == "tool.failed"
    assert payload["ok"] is False


@pytest.mark.asyncio
async def test_upstream_stop_uses_exact_profile_scoped_run(monkeypatch):
    requested = []
    async_client = httpx.AsyncClient

    def handler(request):
        requested.append((request.method, str(request.url), request.headers.get("x-hermes-session-id")))
        return httpx.Response(200, json={"run_id": "upstream-run-7", "status": "stopping"}, request=request)

    monkeypatch.setenv("HERMES_API_URL", "http://hermes.test")
    monkeypatch.setattr(
        hermes_api.httpx,
        "AsyncClient",
        lambda *_args, **_kwargs: async_client(transport=httpx.MockTransport(handler)),
    )

    result = await hermes_api._request_upstream_stop(
        "http://hermes.test",
        "askpdf-run-profile-1",
        "upstream-run-7",
        {"X-Hermes-Session-Id": "session-3"},
    )

    assert requested == [(
        "POST",
        "http://hermes.test/p/askpdf-run-profile-1/v1/runs/upstream-run-7/stop",
        "session-3",
    )]
    assert result["status"] == "stopping"


@pytest.mark.asyncio
async def test_upstream_stop_is_not_confirmed_until_hermes_is_terminal(monkeypatch):
    statuses = iter(("stopping", "running", "cancelled"))
    async_client = httpx.AsyncClient

    def handler(request):
        return httpx.Response(
            200,
            json={"run_id": "upstream-run-7", "status": next(statuses)},
            request=request,
        )

    monkeypatch.setattr(
        hermes_api.httpx,
        "AsyncClient",
        lambda *_args, **_kwargs: async_client(transport=httpx.MockTransport(handler)),
    )

    result = await hermes_api._confirm_upstream_stop(
        "http://hermes.test",
        "askpdf-run-profile-1",
        "upstream-run-7",
        {},
        timeout_seconds=1,
        poll_interval_seconds=0.01,
    )

    assert result == {
        "confirmed": True,
        "status": "cancelled",
        "last_event": None,
    }


@pytest.mark.asyncio
async def test_upstream_stop_reports_unconfirmed_while_executor_is_still_running(monkeypatch):
    async_client = httpx.AsyncClient

    def handler(request):
        return httpx.Response(
            200,
            json={"run_id": "upstream-run-7", "status": "stopping"},
            request=request,
        )

    monkeypatch.setattr(
        hermes_api.httpx,
        "AsyncClient",
        lambda *_args, **_kwargs: async_client(transport=httpx.MockTransport(handler)),
    )

    result = await hermes_api._confirm_upstream_stop(
        "http://hermes.test",
        "askpdf-run-profile-1",
        "upstream-run-7",
        {},
        timeout_seconds=0,
    )

    assert result == {"confirmed": False, "status": "stopping"}


@pytest.mark.asyncio
async def test_upstream_stop_rejects_a_malformed_acknowledgement(monkeypatch):
    async_client = httpx.AsyncClient

    def handler(request):
        return httpx.Response(200, text="not-json", request=request)

    monkeypatch.setattr(
        hermes_api.httpx,
        "AsyncClient",
        lambda *_args, **_kwargs: async_client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(httpx.DecodingError, match="not valid JSON"):
        await hermes_api._request_upstream_stop(
            "http://hermes.test",
            "askpdf-run-profile-1",
            "upstream-run-7",
            {},
        )


def _cancel_payload():
    return {
        "continuation": {
            "binding_type": "hermes_session",
            "payload": {
                "session_id": "session-3",
                "upstream_run_id": "upstream-run-7",
                "runtime_profile": "askpdf-run-profile-1",
            },
        },
    }


def test_cancel_retires_profile_only_after_confirmed_upstream_cancellation(monkeypatch, tmp_path):
    retired = []
    async_client = httpx.AsyncClient

    def handler(request):
        status = "stopping" if request.method == "POST" else "cancelled"
        return httpx.Response(200, json={"status": status}, request=request)

    monkeypatch.setenv("HERMES_API_URL", "http://hermes.test")
    monkeypatch.setenv("HERMES_RUNTIME_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setattr(
        hermes_api.httpx,
        "AsyncClient",
        lambda *_args, **_kwargs: async_client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(
        hermes_api.RunProfileManager,
        "retire",
        lambda _self, profile: retired.append(profile),
    )

    with TestClient(hermes_api.create_app()) as client:
        response = client.post("/v1/runs/run-1/cancel", json=_cancel_payload())

    assert response.json()["result"]["status"] == "cancelled"
    assert retired == ["askpdf-run-profile-1"]


def test_cancel_keeps_profile_when_upstream_stop_is_unconfirmed(monkeypatch, tmp_path):
    retired = []
    async_client = httpx.AsyncClient

    def handler(request):
        return httpx.Response(200, json={"status": "stopping"}, request=request)

    monkeypatch.setenv("HERMES_API_URL", "http://hermes.test")
    monkeypatch.setenv("HERMES_RUNTIME_STATE_PATH", str(tmp_path / "state.json"))
    monkeypatch.setenv("AGENT_RUNTIME_CANCEL_CONFIRM_TIMEOUT_SECONDS", "0.01")
    monkeypatch.setattr(
        hermes_api.httpx,
        "AsyncClient",
        lambda *_args, **_kwargs: async_client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr(
        hermes_api.RunProfileManager,
        "retire",
        lambda _self, profile: retired.append(profile),
    )

    with TestClient(hermes_api.create_app()) as client:
        response = client.post("/v1/runs/run-1/cancel", json=_cancel_payload())

    body = response.json()
    assert body["status"] == "failed"
    assert body["error"]["code"] == "hermes_stop_unconfirmed"
    assert retired == []


@pytest.mark.asyncio
async def test_hermes_controls_use_neutral_contracts(monkeypatch):
    monkeypatch.setenv("COMPOSE_PROFILES", "hermes")
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "32768")
    monkeypatch.setenv("HERMES_MODEL_PROVIDER", "lmstudio")
    calls = []
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")

    async def fake_json(method, path, **kwargs):
        calls.append((method, path, kwargs["json"]))
        return {"accepted": True}

    monkeypatch.setattr(adapter, "_json", fake_json)
    binding = ContinuationBinding("hermes_session", {"session_id": "session-1", "upstream_run_id": "upstream-1"})
    request = AgentRuntimeRequest("run-1", "thread-1", "hermes_rag_agent", "hermes", "hermes_agent", continuation=binding)
    await adapter.respond_to_approval(request, RuntimeApprovalResponse("approve", scope="session"))
    with pytest.raises(RuntimeError) as error:
        await adapter.steer_live(request, RuntimeSteeringInput("Use the newer evidence"))
    assert calls[0][1] == "/v1/runs/run-1/approval"
    assert calls[0][2]["response"] == {"choice": "session", "resolve_all": True}
    assert len(calls) == 1
    assert error.value.code == "runtime_capability_unsupported"
    assert error.value.details["operation_id"] == "run.steer_live"


@pytest.mark.asyncio
async def test_hermes_live_steering_makes_no_transport_request():
    requested = []

    def handler(request):
        requested.append(request)
        return httpx.Response(200, json={"accepted": True}, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test", client=client)
    request = AgentRuntimeRequest("run-1", "thread-1", "hermes_rag_agent", "hermes", "hermes_agent")

    with pytest.raises(RuntimeError) as error:
        await adapter.steer_live(request, RuntimeSteeringInput("focus"))

    assert error.value.code == "runtime_capability_unsupported"
    assert error.value.details["operation_id"] == "run.steer_live"
    assert requested == []
    await client.aclose()


@pytest.mark.asyncio
async def test_hermes_stream_replays_from_last_event_id(monkeypatch):
    monkeypatch.setenv("COMPOSE_PROFILES", "hermes")
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "32768")
    monkeypatch.setenv("HERMES_MODEL_PROVIDER", "lmstudio")
    async def tool_capable(_model):
        return True
    monkeypatch.setattr("app.runtime.hermes_adapter.check_model_can_invoke_tools", tool_capable)
    request = AgentRuntimeRequest("run-1", "thread-1", "hermes_rag_agent", "hermes", "hermes_agent")
    progress = {
        "event_id": "run-1:346",
        "run_id": "run-1",
        "sequence": 346,
        "kind": "output.delta",
        "payload": {"delta": "partial"},
    }
    terminal = {
        "event_id": "run-1:347",
        "run_id": "run-1",
        "sequence": 347,
        "kind": "run.completed",
        "payload": {},
        "terminal": True,
    }
    calls: list[str] = []

    def handler(http_request: httpx.Request) -> httpx.Response:
        calls.append(http_request.method)
        if http_request.method == "POST":
            body = f"id: run-1:346\nevent: output.delta\ndata: {json.dumps({'event': progress})}\n\n"
        else:
            assert http_request.url.params["after_event_id"] == "run-1:346"
            assert "after_sequence" not in http_request.url.params
            body = f"id: run-1:347\nevent: run.completed\ndata: {json.dumps({'event': terminal, 'result': {'status': 'completed', 'output': 'recovered'}})}\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test", client=client)
    result = await adapter.start(
        request,
        context=RuntimeExecutionContext(resolved_spec={"managed_profile": {"model_policy": {"model": "tool-model"}}}),
    )

    assert result.output == "recovered"
    assert calls == ["POST", "GET"]
    await client.aclose()


@pytest.mark.asyncio
async def test_hermes_start_rejects_model_without_native_tool_invocation(monkeypatch):
    monkeypatch.setenv("COMPOSE_PROFILES", "hermes")
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "32768")
    monkeypatch.setenv("HERMES_MODEL_PROVIDER", "lmstudio")
    async def tool_incapable(_model):
        return False
    monkeypatch.setattr("app.runtime.hermes_adapter.check_model_can_invoke_tools", tool_incapable)
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    request = AgentRuntimeRequest("run-1", "thread-1", "hermes_rag_agent", "hermes", "hermes_agent")

    with pytest.raises(RuntimeError) as error:
        await adapter.start(
            request,
            context=RuntimeExecutionContext(resolved_spec={"managed_profile": {"model_policy": {"model": "text-only"}}}),
        )

    assert error.value.code == "runtime_model_tool_calling_unsupported"
    assert error.value.details == {"framework": "hermes", "model": "text-only"}


def test_hermes_runtime_requires_explicit_upstream(monkeypatch, tmp_path):
    monkeypatch.delenv("HERMES_API_URL", raising=False)
    monkeypatch.setenv("HERMES_RUNTIME_STATE_PATH", str(tmp_path / "hermes.json"))
    with pytest.raises(builtins.RuntimeError, match="HERMES_API_URL is required"):
        hermes_api.create_app()


def test_hermes_healthz_is_liveness_only(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_API_URL", "http://unavailable.test")
    monkeypatch.setenv("HERMES_RUNTIME_STATE_PATH", str(tmp_path / "hermes.json"))
    with TestClient(hermes_api.create_app()) as client:
        response = client.get("/healthz")
    assert response.status_code == 200


def test_hermes_file_store_rejects_multiple_workers(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_API_URL", "http://hermes.test")
    monkeypatch.setenv("HERMES_RUNTIME_STATE_PATH", str(tmp_path / "hermes.json"))
    monkeypatch.setenv("HERMES_RUNTIME_STORAGE_BACKEND", "file")
    monkeypatch.setenv("HERMES_RUNTIME_WORKERS", "2")
    with pytest.raises(builtins.RuntimeError, match="one worker only"):
        hermes_api.create_app()


def test_hermes_proof_rejects_non_file_storage(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_API_URL", "http://hermes.test")
    monkeypatch.setenv("HERMES_RUNTIME_STATE_PATH", str(tmp_path / "hermes.json"))
    monkeypatch.setenv("HERMES_RUNTIME_STORAGE_BACKEND", "postgres")
    monkeypatch.setenv("HERMES_RUNTIME_WORKERS", "1")
    with pytest.raises(builtins.RuntimeError, match="PostgreSQL execution storage is not enabled"):
        hermes_api.create_app()


@pytest.mark.asyncio
async def test_hermes_cancel_and_inspect_require_upstream_binding(monkeypatch):
    monkeypatch.setenv("COMPOSE_PROFILES", "hermes")
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "32768")
    monkeypatch.setenv("HERMES_MODEL_PROVIDER", "lmstudio")
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    request = AgentRuntimeRequest("run-1", "thread-1", "hermes_rag_agent", "hermes", "hermes_agent")
    operations = (
        lambda: adapter.cancel(request),
        lambda: adapter.inspect_state(request),
        lambda: adapter.respond_to_approval(request, RuntimeApprovalResponse("once")),
    )
    for operation in operations:
        with pytest.raises(RuntimeError, match="binding"):
            await operation()
    with pytest.raises(RuntimeError) as error:
        await adapter.steer_live(request, RuntimeSteeringInput("focus"))
    assert error.value.code == "runtime_capability_unsupported"


@pytest.mark.asyncio
async def test_hermes_adapter_rejects_missing_context_length(monkeypatch):
    monkeypatch.setenv("COMPOSE_PROFILES", "hermes")
    monkeypatch.delenv("HERMES_MODEL_CONTEXT_LENGTH", raising=False)
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    request = AgentRuntimeRequest("run-1", "thread-1", "hermes_rag_agent", "hermes", "hermes_agent")
    with pytest.raises(RuntimeError, match="context length"):
        await adapter.start(request, context=None)


def _readiness_response(monkeypatch, tmp_path, *, hermes_status, mcp_status=200, mcp_required=True, rendered_context=None):
    requested_urls = []
    async_client = httpx.AsyncClient

    def handler(request):
        requested_urls.append(str(request.url))
        status = hermes_status if request.url.host == "hermes.test" else mcp_status
        return httpx.Response(status, request=request)

    def client_factory(*_args, **_kwargs):
        return async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setenv("HERMES_RUNTIME_STATE_PATH", str(tmp_path / "hermes-readiness.json"))
    context_length = os.environ["HERMES_MODEL_CONTEXT_LENGTH"]
    rendered_config = tmp_path / "hermes-config.yaml"
    rendered_config.write_text(f"model:\n  context_length: {rendered_context or context_length}\n")
    monkeypatch.setenv("HERMES_RENDERED_CONFIG_PATH", str(rendered_config))
    monkeypatch.setenv("HERMES_API_URL", "http://hermes.test")
    monkeypatch.setenv("ASKPDF_MCP_HEALTH_URL", "http://mcp.test/healthz")
    monkeypatch.setenv("ASKPDF_MCP_REQUIRED", "true" if mcp_required else "false")
    monkeypatch.setattr(hermes_api.httpx, "AsyncClient", client_factory)
    with TestClient(hermes_api.create_app()) as client:
        response = client.get("/readyz")
    return response, requested_urls


def test_hermes_readiness_accepts_healthy_hermes_and_mcp(monkeypatch, tmp_path):
    response, requested_urls = _readiness_response(
        monkeypatch, tmp_path, hermes_status=200, mcp_status=204
    )
    assert response.status_code == 200
    assert response.json()["checks"]["hermes"]["status"] == "ok"
    assert response.json()["checks"]["mcp"]["status"] == "ok"
    assert requested_urls == ["http://hermes.test/health", "http://mcp.test/healthz"]


def test_hermes_readiness_rejects_rendered_context_mismatch(monkeypatch, tmp_path):
    configured = int(os.environ["HERMES_MODEL_CONTEXT_LENGTH"])
    response, _ = _readiness_response(
        monkeypatch,
        tmp_path,
        hermes_status=200,
        mcp_status=204,
        rendered_context=configured + 1,
    )
    assert response.status_code == 503
    assert response.json()["checks"]["model_context"] == {
        "status": "failed",
        "configured_context_length": configured,
        "rendered_context_length": configured + 1,
        "provider": os.getenv("HERMES_MODEL_PROVIDER", "custom"),
    }


def test_hermes_readiness_does_not_invent_health_route_from_mcp_transport(monkeypatch, tmp_path):
    monkeypatch.delenv("ASKPDF_MCP_HEALTH_URL", raising=False)
    monkeypatch.setenv("ASKPDF_MCP_URL", "http://mcp.test/internal/mcp/")
    response, requested_urls = _readiness_response(
        monkeypatch, tmp_path, hermes_status=200, mcp_status=200
    )

    # The helper configures the explicit health URL; remove it and exercise the
    # app again to ensure the streamable transport URL is never treated as a
    # conventional GET health endpoint.
    monkeypatch.delenv("ASKPDF_MCP_HEALTH_URL", raising=False)
    async_client = httpx.AsyncClient

    def handler(request):
        requested_urls.append(str(request.url))
        return httpx.Response(200, request=request)

    monkeypatch.setattr(
        hermes_api.httpx,
        "AsyncClient",
        lambda *_args, **_kwargs: async_client(transport=httpx.MockTransport(handler)),
    )
    with TestClient(hermes_api.create_app()) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["checks"]["mcp"]["status"] == "not_checked"
    assert not any("/internal/mcp/healthz" in url for url in requested_urls)


def test_hermes_readiness_rejects_unhealthy_required_mcp(monkeypatch, tmp_path):
    response, requested_urls = _readiness_response(
        monkeypatch, tmp_path, hermes_status=200, mcp_status=503
    )
    assert response.status_code == 503
    assert response.json()["checks"]["hermes"]["status"] == "ok"
    assert response.json()["checks"]["mcp"]["status"] == "failed"
    assert requested_urls == ["http://hermes.test/health", "http://mcp.test/healthz"]


def test_hermes_readiness_rejects_unhealthy_hermes_without_mcp_probe(monkeypatch, tmp_path):
    response, requested_urls = _readiness_response(
        monkeypatch, tmp_path, hermes_status=503, mcp_status=200
    )
    assert response.status_code == 503
    assert response.json()["checks"]["hermes"]["status"] == "failed"
    assert response.json()["checks"]["mcp"]["status"] == "not_checked"
    assert requested_urls == ["http://hermes.test/health"]


def test_hermes_readiness_skips_mcp_when_not_required(monkeypatch, tmp_path):
    response, requested_urls = _readiness_response(
        monkeypatch, tmp_path, hermes_status=200, mcp_status=503, mcp_required=False
    )
    assert response.status_code == 200
    assert response.json()["checks"]["mcp"] == {"status": "ok", "required": False}
    assert requested_urls == ["http://hermes.test/health"]
