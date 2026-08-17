from __future__ import annotations

from fastapi.testclient import TestClient
import time
import pytest
from app.runtime.contracts import AgentRuntimeResult
from runtime_service.execution_store import ExecutionStore

from runtime_service.api import create_app


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
    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["checks"]["checkpoint_store"]["backend"] == "memory"
    assert "DATABASE_URL" not in response.text


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
