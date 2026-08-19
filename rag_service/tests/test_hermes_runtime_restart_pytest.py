from __future__ import annotations

import os
import uuid

import httpx
import pytest

from hermes_runtime.execution_store import HermesExecutionStore

from hermes_test_helpers import RUNTIME_URL, read_sse, runtime_payload


pytestmark = pytest.mark.skipif(
    os.getenv("PHASE7_HERMES_INTEGRATION", "").lower() not in {"1", "true", "yes", "on"},
    reason="requires the Phase 7 Hermes integration Compose profile",
)


RECOVERY_RUN_ID = os.getenv("PHASE7_RECOVERY_RUN_ID") or f"phase7-recovery-{uuid.uuid4().hex}"


@pytest.mark.asyncio
async def test_seed_restart_recovery_record() -> None:
    payload = runtime_payload(RECOVERY_RUN_ID, "Work slowly and continue until stopped.")
    headers = {"authorization": f"Bearer {os.environ['HERMES_API_TOKEN']}"}
    async with httpx.AsyncClient(base_url=os.environ["HERMES_API_URL"], headers=headers) as hermes:
        response = await hermes.post(
            "/v1/runs",
            json={
                "input": payload["request"]["input"]["question"],
                "instructions": "Respond one word at a time.",
                "model": "phase5-deterministic",
                "provider": "custom",
                "metadata": {"askpdf_run_id": RECOVERY_RUN_ID},
            },
        )
        response.raise_for_status()
        upstream = response.json()
        status = await hermes.get(f"/v1/runs/{upstream['run_id']}")
        status.raise_for_status()
        upstream_status = status.json()
    binding = {
        "binding_type": "hermes_session",
        "binding_version": 1,
        "runtime_version": "hermes-gateway-1",
        "payload": {"upstream_run_id": upstream["run_id"], "session_id": upstream_status["session_id"]},
    }
    store = HermesExecutionStore(os.environ["HERMES_RUNTIME_STATE_PATH"])
    store.update(RECOVERY_RUN_ID, status="running", payload=payload, continuation=binding)


@pytest.mark.asyncio
async def test_recovered_run_reconnects_without_another_upstream_start() -> None:
    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=30) as client:
        events = await read_sse(client, "GET", f"/v1/runs/{RECOVERY_RUN_ID}/events")
    assert any(item["event"].get("terminal") for item in events)
