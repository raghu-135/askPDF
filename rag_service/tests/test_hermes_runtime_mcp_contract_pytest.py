import os
from pathlib import Path

from fastapi.testclient import TestClient

from hermes_runtime.api import create_app


def _payload(allowed_tools):
    return {
        "definition": {"framework": "hermes", "builder_id": "hermes_agent"},
        "spec": {
            "schema_version": 2,
            "definition_version": 1,
            "config": {
                "mcp_server": "askpdf",
                "allowed_tool_ids": allowed_tools,
                "system_prompt": "Use document evidence.",
            },
        },
    }


def test_hermes_accepts_allowlisted_mcp_document_tool(monkeypatch):
    monkeypatch.setenv("HERMES_API_URL", "http://hermes.test")
    monkeypatch.setenv("HERMES_MCP_ALLOWED_TOOLS", "document_evidence,clarify_intent")
    with TestClient(create_app()) as client:
        response = client.post("/v1/validate", json=_payload(["document_evidence"]))
    assert response.status_code == 200
    assert response.json()["result"]["validation"]["valid"] is True


def test_hermes_rejects_tool_outside_allowlist(monkeypatch):
    monkeypatch.setenv("HERMES_API_URL", "http://hermes.test")
    monkeypatch.setenv("HERMES_MCP_ALLOWED_TOOLS", "document_evidence")
    with TestClient(create_app()) as client:
        response = client.post("/v1/validate", json=_payload(["admin_delete_everything"]))
    validation = response.json()["result"]["validation"]
    assert validation["valid"] is False
    assert any(issue["code"] == "unsupported_tool_allowlist" for issue in validation["issues"])


def test_phase7_runner_enables_and_guards_every_integration_proof_command():
    repository = Path(os.getenv("ASKPDF_REPO_DIR", "/workspace"))
    script = (repository / "run_tests.sh").read_text()
    phase7 = script.split('if [ "${RUN_PHASE7:-0}" = "1" ]; then', 1)[1].split("\nfi", 1)[0]

    assert phase7.count("-e PHASE7_HERMES_INTEGRATION=true") == 2
    assert phase7.count("-e ASKPDF_FAIL_IF_ALL_SKIPPED=true") == 2
    assert "hermes-fake" not in phase7
    assert "test_real_hermes_container_smoke_pytest.py" in phase7
    assert " hermes hermes-runtime" in phase7
