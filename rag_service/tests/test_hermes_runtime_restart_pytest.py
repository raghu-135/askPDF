from __future__ import annotations

import os

import httpx
import pytest

from hermes_runtime.execution_store import HermesExecutionStore

from test_hermes_runtime_integration_pytest import _payload, _sse, FAKE_URL, RUNTIME_URL


@pytest.mark.asyncio
async def test_seed_restart_recovery_record() -> None:
    fake_payload = _payload("phase7-recovery")
    async with httpx.AsyncClient(base_url=FAKE_URL) as fake:
        response = await fake.post("/v1/runs", json={"metadata": {"askpdf_run_id": "phase7-recovery"}})
        response.raise_for_status()
        upstream = response.json()
    binding = {
        "binding_type": "hermes_session",
        "binding_version": 1,
        "runtime_version": "hermes-gateway-1",
        "payload": {"upstream_run_id": upstream["run_id"], "session_id": upstream["session_id"]},
    }
    store = HermesExecutionStore(os.environ["HERMES_RUNTIME_STATE_PATH"])
    store.update("phase7-recovery", status="running", payload=fake_payload, continuation=binding)


@pytest.mark.asyncio
async def test_recovered_run_reconnects_without_another_upstream_start() -> None:
    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=30) as client:
        events = await _sse(client, "GET", "/v1/runs/phase7-recovery/events")
    assert any(item["event"].get("terminal") for item in events)
    async with httpx.AsyncClient(base_url=FAKE_URL) as fake:
        state = (await fake.get("/debug/state")).json()
    matching = [value for value in state["runs"].values() if value["payload"].get("metadata", {}).get("askpdf_run_id") == "phase7-recovery"]
    assert len(matching) == 1
