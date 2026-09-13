from __future__ import annotations

import os


def test_product_api_requires_admin_token(api_client):
    api_client.headers["Authorization"] = f"Bearer {os.environ['ASKPDF_ADMIN_TOKEN']}"
    response = api_client.get("/api/threads")
    assert response.status_code == 200

    api_client.headers.pop("Authorization")
    response = api_client.get("/api/threads")
    assert response.status_code == 401

    for authorization in ("Bearer wrong-token", "Basic test-control-plane-token-32-characters", "Bearer  test-control-plane-token-32-characters"):
        response = api_client.get("/api/threads", headers={"Authorization": authorization})
        assert response.status_code == 401


def test_trusted_proxy_authentication_is_opt_in(api_client, monkeypatch):
    api_client.headers.pop("Authorization")
    response = api_client.get("/api/threads", headers={"X-Authenticated-User": "proxy-user"})
    assert response.status_code == 401

    monkeypatch.setenv("ASKPDF_TRUST_PROXY_AUTH", "true")
    response = api_client.get("/api/threads", headers={"X-Authenticated-User": "proxy-user"})
    assert response.status_code == 200


def test_health_endpoints_remain_public(api_client):
    api_client.headers.pop("Authorization")
    assert api_client.get("/health").status_code in {200, 503}
    assert api_client.get("/ready").status_code in {200, 503}


def test_mcp_runtime_health_requires_langgraph_runtime_token(api_client, monkeypatch):
    api_client.headers.pop("Authorization")
    assert api_client.get("/internal/mcp/health").status_code == 401
    assert api_client.get(
        "/internal/mcp/health",
        headers={"Authorization": "Bearer wrong-token"},
    ).status_code == 401

    token = "test-langgraph-runtime-token-32-characters"
    monkeypatch.setenv("LANGGRAPH_RUNTIME_TOKEN", token)
    response = api_client.get("/internal/mcp/health", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert any(tool["name"] == "get_thread_shape" for tool in response.json()["tools"])
