from __future__ import annotations

from fastapi.testclient import TestClient

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