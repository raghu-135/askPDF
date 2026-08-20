import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fastapi import HTTPException

from hermes_runtime.api import _error, _HermesEventBudget, _upstream_timeout, create_app


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


def test_hermes_reports_frozen_profile_tool_allowlist(monkeypatch):
    monkeypatch.setenv("HERMES_API_URL", "http://hermes.test")
    with TestClient(create_app()) as client:
        response = client.post("/v1/validate", json=_payload(["search_documents", "get_thread_shape"]))
    assert response.status_code == 200
    validation = response.json()["result"]["validation"]
    assert validation["valid"] is True
    assert validation["runtime_metadata"]["allowed_tool_ids"] == ["get_thread_shape", "search_documents"]


def test_environment_cannot_override_frozen_profile_tool_allowlist(monkeypatch):
    monkeypatch.setenv("HERMES_API_URL", "http://hermes.test")
    monkeypatch.setenv("HERMES_MCP_ALLOWED_TOOLS", "admin_delete_everything")
    with TestClient(create_app()) as client:
        response = client.post("/v1/validate", json=_payload(["search_documents"]))
    validation = response.json()["result"]["validation"]
    assert validation["valid"] is True
    assert validation["runtime_metadata"]["allowed_tool_ids"] == ["search_documents"]


def test_stream_timeout_uses_frozen_task_duration_not_generic_read_timeout(monkeypatch):
    monkeypatch.setenv("HERMES_RUNTIME_READ_TIMEOUT_SECONDS", "30")

    assert _upstream_timeout().read == 30
    assert _upstream_timeout(300).read == 300


def test_runtime_errors_preserve_safe_message_and_sanitized_details():
    error = _error(
        "hermes_upstream_timeout",
        "Hermes did not produce an event before the task execution timeout",
        retryable=True,
        details={"phase": "event_stream", "error_type": "ReadTimeout"},
    )

    assert error["safe_message"]
    assert error["details"] == {"phase": "event_stream", "error_type": "ReadTimeout"}


def test_message_deltas_use_output_budget_not_lifecycle_event_budget():
    budget = _HermesEventBudget(max_lifecycle_events=2, max_output_chars=1_000)

    for _ in range(300):
        budget.observe("output.delta", "abc")
    budget.observe("tool.started")
    budget.observe("run.completed")

    assert budget.details() == {
        "lifecycle_event_count": 2,
        "output_char_count": 900,
        "raw_frame_count": 302,
    }


def test_empty_delta_flood_still_consumes_lifecycle_budget():
    budget = _HermesEventBudget(max_lifecycle_events=2, max_output_chars=100)

    budget.observe("output.delta", "")
    budget.observe("output.delta", "")
    with pytest.raises(HTTPException, match="lifecycle"):
        budget.observe("output.delta", "")


def test_phase7_runner_enables_and_guards_every_integration_proof_command():
    repository = Path(os.getenv("ASKPDF_REPO_DIR", "/workspace"))
    script = (repository / "run_tests.sh").read_text()
    phase7 = script.split('if [ "${RUN_PHASE7:-0}" = "1" ]; then', 1)[1].split("\nfi", 1)[0]

    assert phase7.count("-e PHASE7_HERMES_INTEGRATION=true") == 2
    assert phase7.count("-e ASKPDF_FAIL_IF_ALL_SKIPPED=true") == 2
    assert "hermes-fake" not in phase7
    assert "test_real_hermes_container_smoke_pytest.py" in phase7
    assert " hermes hermes-runtime" in phase7
