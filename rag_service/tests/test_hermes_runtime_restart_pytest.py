from __future__ import annotations

import os
import uuid

import httpx
import pytest

from hermes_runtime.execution_store import HermesExecutionStore

from test_hermes_runtime_integration_pytest import _payload, _sse, FAKE_URL, RUNTIME_URL


pytestmark = pytest.mark.skipif(
    os.getenv("PHASE7_HERMES_INTEGRATION", "").lower() not in {"1", "true", "yes", "on"},
    reason="requires the Phase 7 Hermes integration Compose profile",
)


RECOVERY_RUN_ID = os.getenv("PHASE7_RECOVERY_RUN_ID") or f"phase7-recovery-{uuid.uuid4().hex}"


@pytest.mark.asyncio
async def test_seed_restart_recovery_record() -> None:
    fake_payload = _payload(RECOVERY_RUN_ID)
    async with httpx.AsyncClient(base_url=FAKE_URL) as fake:
        response = await fake.post("/v1/runs", json={"metadata": {"askpdf_run_id": RECOVERY_RUN_ID}})
        response.raise_for_status()
        upstream = response.json()
    binding = {
        "binding_type": "hermes_session",
        "binding_version": 1,
        "runtime_version": "hermes-gateway-1",
        "payload": {"upstream_run_id": upstream["run_id"], "session_id": upstream["session_id"]},
    }
    store = HermesExecutionStore(os.environ["HERMES_RUNTIME_STATE_PATH"])
    store.update(RECOVERY_RUN_ID, status="running", payload=fake_payload, continuation=binding)


@pytest.mark.asyncio
async def test_recovered_run_reconnects_without_another_upstream_start() -> None:
    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=30) as client:
        events = await _sse(client, "GET", f"/v1/runs/{RECOVERY_RUN_ID}/events")
    assert any(item["event"].get("terminal") for item in events)
    async with httpx.AsyncClient(base_url=FAKE_URL) as fake:
        state = (await fake.get("/debug/state")).json()
    matching = [value for value in state["runs"].values() if value["payload"].get("metadata", {}).get("askpdf_run_id") == RECOVERY_RUN_ID]
    assert len(matching) == 1
