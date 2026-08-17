import builtins
import httpx
import pytest
from fastapi.testclient import TestClient

from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest, ContinuationBinding
from app.runtime.hermes_adapter import HermesRuntimeAdapter
from app.runtime.errors import RuntimeError
from hermes_runtime import api as hermes_api


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
    adapter = HermesRuntimeAdapter(base_url="http://hermes.test")
    request = AgentRuntimeRequest("run-1", "thread-1", "hermes_rag_agent", "hermes", "hermes_agent")
    for operation in (adapter.cancel, adapter.inspect):
        with pytest.raises(RuntimeError, match="binding"):
            await operation(request)


def _readiness_response(monkeypatch, tmp_path, *, hermes_status, mcp_status=200, mcp_required=True):
    requested_urls = []
    async_client = httpx.AsyncClient

    def handler(request):
        requested_urls.append(str(request.url))
        status = hermes_status if request.url.host == "hermes.test" else mcp_status
        return httpx.Response(status, request=request)

    def client_factory(*_args, **_kwargs):
        return async_client(transport=httpx.MockTransport(handler))

    monkeypatch.setenv("HERMES_RUNTIME_STATE_PATH", str(tmp_path / "hermes-readiness.json"))
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
