from __future__ import annotations

from fastapi.testclient import TestClient
import httpx
import time
import pytest
from app.runtime.contracts import AgentRuntimeResult
from runtime_service.execution_store import ExecutionStore

from runtime_service.api import create_app
from runtime_service.dependencies import probe_mcp, probe_provider


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 406, 422, 500])
async def test_mcp_readiness_rejects_http_errors(status):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(status)))
    try:
        result = await probe_mcp("http://mcp/internal/mcp/", 1, client=client)
    finally:
        await client.aclose()
    assert result == {"ok": False, "http_status": status, "reason": "unexpected_status"}


@pytest.mark.asyncio
async def test_mcp_readiness_requires_a_valid_tools_list():
    response = {"jsonrpc": "2.0", "id": "runtime-readiness", "result": {"tools": [{"name": "get_thread_shape"}]}}
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=response)))
    try:
        result = await probe_mcp("http://mcp/internal/mcp/", 1, client=client)
    finally:
        await client.aclose()
    assert result == {"ok": True, "http_status": 200, "protocol": "mcp", "capability_ids": ["get_thread_shape"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 406, 422, 500])
async def test_provider_readiness_rejects_http_errors(status):
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(status)))
    try:
        result = await probe_provider("http://provider/v1", 1, client=client)
    finally:
        await client.aclose()
    assert result == {"ok": False, "http_status": status, "reason": "unexpected_status"}


@pytest.mark.asyncio
async def test_provider_readiness_requires_a_models_list():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"data": []})))
    try:
        result = await probe_provider("http://provider/v1", 1, client=client)
    finally:
        await client.aclose()
    assert result == {"ok": True, "http_status": 200, "capability_ids": []}


def test_runtime_healthz_is_liveness_only(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "")
    monkeypatch.setenv("LLM_API_URL", "")
    with TestClient(create_app()) as client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "langgraph-runtime"}


def test_runtime_readyz_is_structured_when_optional_probes_are_unconfigured(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "")
    monkeypatch.setenv("LLM_API_URL", "")
    with TestClient(create_app()) as client:
        response = client.get("/readyz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["checks"]["checkpoint_store"]["backend"] == "memory"
    assert payload["checks"]["execution_store"]["status"] == "ok"
    assert "mcp" not in payload["checks"]
    assert "provider" not in payload["checks"]
    assert "DATABASE_URL" not in response.text


def test_runtime_startup_and_dependency_endpoints_are_separate(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "")
    monkeypatch.setenv("LLM_API_URL", "")
    with TestClient(create_app()) as client:
        assert client.get("/startupz").json() == {"status": "ok"}
        dependency_response = client.get("/v1/dependencies")
    dependencies = dependency_response.json()["result"]["dependencies"]
    assert dependencies["mcp"]["state"] == "not_configured"
    assert dependencies["provider"]["state"] == "not_configured"


def test_dependency_outage_does_not_change_readiness_but_blocks_required_run(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "http://unavailable/mcp")
    monkeypatch.setenv("LLM_API_URL", "")

    async def unavailable_probe(*_args, **_kwargs):
        return {"ok": False, "reason": "ConnectError"}

    monkeypatch.setattr("runtime_service.dependencies.probe_mcp", unavailable_probe)
    payload = {
        "request": {
            "run_id": "dependency-blocked",
            "thread_id": "thread-1",
            "definition_id": "router_rag_agent",
            "framework": "langgraph",
            "builder_id": "langgraph_graph",
            "input": {"question": "hello"},
            "options": {},
        },
        "context": {"resolved_spec": {"config": {"allowed_tool_ids": ["document_evidence"]}}},
    }
    with TestClient(create_app()) as client:
        assert client.get("/readyz").status_code == 200
        response = client.post("/v1/runs/start", json=payload)
        assert client.post("/v1/runs/dependency-blocked/cancel", json={"request": payload["request"]}).status_code == 200
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "runtime_dependency_unavailable"
    assert error["retryable"] is True
    assert error["details"]["dependency"] == "mcp"
    assert "http://unavailable" not in response.text


def test_legacy_strict_readiness_can_gate_on_dependencies(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "http://unavailable/mcp")
    monkeypatch.setenv("LLM_API_URL", "")
    monkeypatch.setenv("AGENT_RUNTIME_LEGACY_STRICT_READINESS", "true")

    async def unavailable_probe(*_args, **_kwargs):
        return {"ok": False, "reason": "ConnectError"}

    monkeypatch.setattr("runtime_service.dependencies.probe_mcp", unavailable_probe)
    with TestClient(create_app()) as client:
        response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["checks"]["mcp"]["state"] == "unavailable"


def test_recovery_loop_reclaims_a_lease_after_restart(monkeypatch):
    monkeypatch.setenv("ASKPDF_AGENT_CHECKPOINTER", "memory")
    monkeypatch.setenv("MCP_LOOPBACK_URL", "")
    monkeypatch.setenv("LLM_API_URL", "")
    monkeypatch.setenv("AGENT_RUNTIME_RECOVERY_LOOP_ENABLED", "true")
    monkeypatch.setenv("AGENT_RUNTIME_RECOVERY_INTERVAL_SECONDS", "1")

    class FakeAdapter:
        async def start(self, request, *, context, event_sink=None):
            return AgentRuntimeResult(status="completed", output={"answer": "recovered"})

    monkeypatch.setattr("app.runtime.langgraph_adapter.LangGraphRuntimeAdapter", FakeAdapter)
    store = ExecutionStore()
    request = {
        "run_id": "restart-recovery",
        "thread_id": "thread-1",
        "definition_id": "router_rag_agent",
        "framework": "langgraph",
        "builder_id": "langgraph_graph",
        "input": {"question": "hello"},
        "options": {},
    }

    async def seed():
        await store.create("restart-recovery", "start", request, {"request": request, "context": {}})
        await store.claim("restart-recovery", owner_id="old-worker", lease_seconds=1)

    import asyncio
    asyncio.run(seed())
    with TestClient(create_app(execution_store=store)):
        time.sleep(2.2)
        record = store._records["restart-recovery"]
        assert record.status == "completed"
