from fastapi.testclient import TestClient

from hermes_runtime.api import create_app


def _payload(allowed_tools):
    return {
        "definition": {"framework": "hermes", "builder_id": "hermes_agent"},
        "spec": {
            "schema_version": 2,
            "config": {
                "mcp_server": "askpdf",
                "allowed_tool_ids": allowed_tools,
                "system_prompt": "Use document evidence.",
            },
        },
    }


def test_hermes_accepts_allowlisted_mcp_document_tool(monkeypatch):
    monkeypatch.setenv("HERMES_MCP_ALLOWED_TOOLS", "document_evidence,clarify_intent")
    with TestClient(create_app()) as client:
        response = client.post("/v1/validate", json=_payload(["document_evidence"]))
    assert response.status_code == 200
    assert response.json()["result"]["validation"]["valid"] is True


def test_hermes_rejects_tool_outside_allowlist(monkeypatch):
    monkeypatch.setenv("HERMES_MCP_ALLOWED_TOOLS", "document_evidence")
    with TestClient(create_app()) as client:
        response = client.post("/v1/validate", json=_payload(["admin_delete_everything"]))
    validation = response.json()["result"]["validation"]
    assert validation["valid"] is False
    assert any(issue["code"] == "unsupported_tool_allowlist" for issue in validation["issues"])
