from __future__ import annotations

import os
import json
from pathlib import Path
import uuid

import httpx
import pytest

from hermes_runtime.execution_store import HermesExecutionStore

from test_hermes_runtime_integration_pytest import _payload, _sse, FAKE_URL, RUNTIME_URL


RECOVERY_RUN_ID = os.getenv("PHASE7_RECOVERY_RUN_ID") or f"phase7-recovery-{uuid.uuid4().hex}"


def _gateway_frame(run_id: str, sequence: int, kind: str, *, source_event_id: str | None = None, terminal: bool = False, result: dict | None = None) -> str:
    event = {"event_id": f"{run_id}:{sequence}", "run_id": run_id, "sequence": sequence, "kind": kind, "payload": {}, "terminal": terminal}
    if source_event_id is not None:
        event["source_event_id"] = source_event_id
    body = {"event": event}
    if result is not None:
        body["result"] = result
    return f"id: {event['event_id']}\nevent: {kind}\ndata: {json.dumps(body)}\n\n"


def test_hermes_store_preserves_sequence_and_terminal_result_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "hermes.json"
    store = HermesExecutionStore(str(path))
    store.create("run-sequence", {"request": {"run_id": "run-sequence"}})
    store.append("run-sequence", _gateway_frame("run-sequence", 1, "output.delta", source_event_id="upstream-1"))
    store.append("run-sequence", _gateway_frame("run-sequence", 2, "output.delta", source_event_id="upstream-2"))
    store.append("run-sequence", _gateway_frame("run-sequence", 3, "run.completed", source_event_id="upstream-3", terminal=True, result={"status": "completed", "output": "ok"}))

    reloaded = HermesExecutionStore(str(path))
    record = reloaded.records["run-sequence"]
    assert record["next_sequence"] == 4
    assert record["terminal_event_id"] == "run-sequence:3"
    assert record["terminal_result"]["output"] == "ok"
    assert reloaded.append("run-sequence", _gateway_frame("run-sequence", 4, "run.completed", source_event_id="upstream-3", terminal=True, result={"status": "completed", "output": "ok"})) is False


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
