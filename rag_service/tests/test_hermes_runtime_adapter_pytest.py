import builtins
import json
import os
from pathlib import Path
import httpx
import pytest
from fastapi.testclient import TestClient

from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest, ContinuationBinding, RuntimeApprovalResponse, RuntimeSteeringInput
from app.runtime.hermes_adapter import HermesRuntimeAdapter
from app.runtime.errors import RuntimeError
from hermes_runtime import api as hermes_api
from hermes_runtime.compatibility import HERMES_REVISION


@pytest.mark.asyncio
async def test_hermes_adapter_has_independent_identity():
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    assert adapter.framework == "hermes"
    assert adapter.builder_id == "hermes_agent"


@pytest.mark.asyncio
async def test_hermes_resume_is_explicitly_unsupported():
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    request = AgentRuntimeRequest("run-1", "thread-1", "hermes_rag_agent", "hermes", "hermes_agent")
    with pytest.raises(RuntimeError, match="resume"):
        await adapter.resume(request, interrupt={}, context=None)


def test_hermes_continuation_binding_is_opaque():
    binding = ContinuationBinding("hermes_session", {"session_id": "session-1"})
    assert binding.to_dict()["payload"]["session_id"] == "session-1"


@pytest.mark.parametrize(
    ("upstream", "neutral"),
    [
        ("message.delta", "output.delta"),
        ("tool.started", "tool.started"),
        ("tool.completed", "tool.completed"),
        ("reasoning.available", "reasoning.available"),
        ("approval.request", "approval.request"),
        ("run.completed", "run.completed"),
        ("run.failed", "run.failed"),
        ("run.cancelled", "run.cancelled"),
        ("subagent.start", "subagent.start"),
        ("subagent.complete", "subagent.complete"),
        ("future.event", "runtime.event"),
    ],
)
def test_pinned_hermes_event_mapping(upstream, neutral):
    assert hermes_api._hermes_event_kind("message", {"event": upstream}) == neutral


def test_checked_in_event_fixtures_are_data_only_and_match_the_pin():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "hermes" / "run_events.json").read_text())
    assert fixture["hermes_revision"] == HERMES_REVISION
    upstream_events = {event["event"] for values in fixture.values() if isinstance(values, list) for event in values}
    assert {"message.delta", "tool.started", "tool.completed", "reasoning.available", "approval.request", "run.completed", "run.failed", "run.cancelled", "subagent.start", "subagent.complete", "run.steered", "future.event"} <= upstream_events


@pytest.mark.asyncio
async def test_hermes_controls_use_neutral_contracts(monkeypatch):
    monkeypatch.setenv("HERMES_RUNTIME_ENABLED", "true")
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
    await adapter.respond_to_approval(request, RuntimeApprovalResponse("session", resolve_all=True))
    await adapter.steer(request, RuntimeSteeringInput("Use the newer evidence"))
    assert calls[0][1] == "/v1/runs/run-1/approval"
    assert calls[0][2]["response"] == {"choice": "session", "resolve_all": True}
    assert calls[1][1] == "/v1/runs/run-1/steer"
    assert calls[1][2]["steering"] == {"text": "Use the newer evidence"}


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
    monkeypatch.setenv("HERMES_RUNTIME_ENABLED", "true")
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "32768")
    monkeypatch.setenv("HERMES_MODEL_PROVIDER", "lmstudio")
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    request = AgentRuntimeRequest("run-1", "thread-1", "hermes_rag_agent", "hermes", "hermes_agent")
    operations = (
        lambda: adapter.cancel(request),
        lambda: adapter.inspect(request),
        lambda: adapter.respond_to_approval(request, RuntimeApprovalResponse("once")),
        lambda: adapter.steer(request, RuntimeSteeringInput("focus")),
    )
    for operation in operations:
        with pytest.raises(RuntimeError, match="binding"):
            await operation()


@pytest.mark.asyncio
async def test_hermes_adapter_rejects_missing_context_length(monkeypatch):
    monkeypatch.setenv("HERMES_RUNTIME_ENABLED", "true")
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
