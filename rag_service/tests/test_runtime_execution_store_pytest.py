from __future__ import annotations

from types import SimpleNamespace

import pytest

from runtime_service.execution_store import ExecutionStore, _json_safe


@pytest.mark.asyncio
async def test_continuation_probe_can_be_replaced_by_start() -> None:
    store = ExecutionStore()

    await store.create("run-1", "continue_run", {"run_id": "run-1"}, {"request": {"run_id": "run-1"}})
    await store.set_status("run-1", "no_continuation")
    await store.append(
        "run-1",
        {"event_id": "run-1:terminal", "kind": "run.continuation_empty", "terminal": True, "payload": {}},
    )

    record = await store.create("run-1", "start", {"run_id": "run-1"}, {"request": {"run_id": "run-1"}})

    assert record.operation == "start"
    assert record.status == "queued"
    assert await store.events_after("run-1") == []


@pytest.mark.asyncio
async def test_failed_start_can_be_retried_with_the_same_run_id() -> None:
    store = ExecutionStore()

    await store.create("run-2", "start", {"run_id": "run-2"}, {"request": {"run_id": "run-2"}})
    await store.set_status("run-2", "failed")
    await store.append(
        "run-2",
        {"event_id": "run-2:terminal", "kind": "run.failed", "terminal": True, "payload": {}},
    )

    record = await store.create("run-2", "start", {"run_id": "run-2"}, {"request": {"run_id": "run-2"}})

    assert record.status == "queued"
    assert await store.events_after("run-2") == []


def test_json_safe_converts_legacy_runtime_objects() -> None:
    value = _json_safe(
        {
            "run": SimpleNamespace(id="run-1"),
            "items": [SimpleNamespace(value=1)],
        }
    )

    assert value == {"run": {"id": "run-1"}, "items": [{"value": 1}]}
