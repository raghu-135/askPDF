from __future__ import annotations

import json
import os
import uuid
from typing import Any

import httpx
import pytest

from app.runtime.contracts import AgentDefinition
from hermes_runtime.api import _recovery_payload, _response_session_id


RUNTIME_URL = os.getenv("HERMES_RUNTIME_URL", "http://hermes-runtime:8200")
FAKE_URL = os.getenv("HERMES_FAKE_URL", "http://hermes-fake:8000")


def _payload(run_id: str) -> dict[str, Any]:
    return {
        "request": {
            "run_id": run_id,
            "thread_id": f"thread-{run_id}",
            "definition_id": "hermes_rag_agent",
            "framework": "hermes",
            "builder_id": "hermes_agent",
            "input": {"question": "deterministic proof"},
            "options": {"llm_model": "fake-hermes-model"},
        },
        "context": {
            "resolved_spec": {
                "config": {
                    "mcp_server": "askpdf",
                    "allowed_tool_ids": ["document_evidence", "clarify_intent"],
                    "system_prompt": "Use approved tools.",
                }
            }
        },
    }


async def _sse(client: httpx.AsyncClient, method: str, path: str, **kwargs: Any) -> list[dict[str, Any]]:
    async with client.stream(method, path, **kwargs) as response:
        assert response.status_code == 200, await response.aread()
        body = await response.aread()
    values = []
    for block in body.decode().split("\n\n"):
        line = next((line for line in block.splitlines() if line.startswith("data:")), None)
        if line:
            values.append(json.loads(line[5:].strip()))
    return values


async def _set_mode(mode: str) -> None:
    async with httpx.AsyncClient(base_url=FAKE_URL) as client:
        response = await client.post("/debug/mode", json={"mode": mode})
        response.raise_for_status()


@pytest.mark.asyncio
async def test_normal_completion_has_one_terminal_event_and_persists_session_binding() -> None:
    await _set_mode("normal")
    run_id = f"phase7-normal-{uuid.uuid4().hex}"
    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=30) as client:
        events = await _sse(client, "POST", "/v1/runs/start", json=_payload(run_id))
        replay = await _sse(client, "GET", f"/v1/runs/{run_id}/events")

    terminals = [item for item in events if item["event"].get("terminal")]
    assert len(terminals) == 1
    binding = terminals[0]["event"]["continuation"]["payload"]
    assert binding["upstream_run_id"]
    assert binding["session_id"]
    assert len([item for item in replay if item["event"].get("terminal")]) == 1


@pytest.mark.asyncio
async def test_unterminated_terminal_frame_is_processed_without_duplicate_completion() -> None:
    await _set_mode("unterminated")
    run_id = f"phase7-unterminated-{uuid.uuid4().hex}"
    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=30) as client:
        events = await _sse(client, "POST", "/v1/runs/start", json=_payload(run_id))
    terminals = [item for item in events if item["event"].get("terminal")]
    assert len(terminals) == 1
    assert terminals[0]["event"]["kind"] == "run.completed"


@pytest.mark.asyncio
async def test_missing_upstream_terminal_becomes_protocol_failure() -> None:
    await _set_mode("missing_terminal")
    run_id = f"phase7-missing-terminal-{uuid.uuid4().hex}"
    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=30) as client:
        events = await _sse(client, "POST", "/v1/runs/start", json=_payload(run_id))
    terminals = [item for item in events if item["event"].get("terminal")]
    assert len(terminals) == 1
    assert terminals[0]["event"]["kind"] == "run.failed"
    assert terminals[0]["result"]["error"]["code"] == "hermes_upstream_protocol_error"


@pytest.mark.asyncio
async def test_mcp_configuration_and_session_run_headers_reach_upstream() -> None:
    await _set_mode("normal")
    run_id = f"phase7-mcp-{uuid.uuid4().hex}"
    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=30) as client:
        await _sse(client, "POST", "/v1/runs/start", json=_payload(run_id))
    async with httpx.AsyncClient(base_url=FAKE_URL) as fake:
        state = (await fake.get("/debug/state")).json()
    matching = [value for value in state["runs"].values() if value["payload"].get("metadata", {}).get("askpdf_run_id") == run_id]
    assert len(matching) == 1
    record = matching[0]
    assert record["payload"]["mcp_servers"]["askpdf"]["tools"]["include"] == ["document_evidence", "clarify_intent"]
    assert record["headers"]["x-hermes-session-id"]
    assert record["headers"]["x-hermes-run-id"]


@pytest.mark.asyncio
async def test_cancel_and_inspect_use_the_persisted_binding() -> None:
    await _set_mode("normal")
    run_id = f"phase7-control-{uuid.uuid4().hex}"
    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=30) as client:
        events = await _sse(client, "POST", "/v1/runs/start", json=_payload(run_id))
    binding = next(item["event"]["continuation"] for item in events if item["event"].get("terminal"))
    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=30) as client:
        inspect = await client.post(f"/v1/runs/{run_id}/inspect", json={"request": _payload(run_id)["request"], "continuation": binding})
        cancel = await client.post(f"/v1/runs/{run_id}/cancel", json={"request": _payload(run_id)["request"], "continuation": binding})
    assert inspect.status_code == 200
    assert cancel.status_code == 200
    assert inspect.json()["result"]["upstream_run_id"]


def test_session_id_precedence_and_recovery_payload_injection() -> None:
    assert _response_session_id({"session_id": "direct", "session": {"id": "nested"}}) == "direct"
    assert _response_session_id({"session": {"id": "nested"}}) == "nested"
    record = {
        "payload": {"request": {"run_id": "run-1", "input": {"question": "q"}}},
        "continuation": {"binding_type": "hermes_session", "payload": {"upstream_run_id": "up-1", "session_id": "s-1"}},
    }
    recovered = _recovery_payload(record)
    assert recovered["request"]["continuation"]["payload"]["upstream_run_id"] == "up-1"
    assert record["payload"]["request"].get("continuation") is None

