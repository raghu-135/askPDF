from __future__ import annotations

import os
import uuid
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


@pytest.mark.asyncio
async def test_resume_creates_a_new_attempt_without_replaying_prior_terminal_events() -> None:
    store = ExecutionStore()
    await store.create("run-resume", "start", {"run_id": "run-resume"}, {"request": {"run_id": "run-resume"}})
    binding = {"binding_type": "checkpoint", "payload": {"id": "cp-1"}}
    await store.append(
        "run-resume",
        {"event_id": "run-resume:terminal", "kind": "run.completed", "terminal": True, "continuation": binding},
    )
    await store.set_status("run-resume", "completed")

    resumed = await store.create(
        "run-resume",
        "resume",
        {"run_id": "run-resume"},
        {"request": {"run_id": "run-resume"}},
    )
    assert resumed.attempt == 2
    assert await store.events_after("run-resume") == []

    await store.append(
        "run-resume",
        {"event_id": "run-resume:terminal", "kind": "run.completed", "terminal": True},
    )
    current = await store.events_after("run-resume")
    all_events = await store.events_after("run-resume", attempt=1) + current
    assert len(current) == 1
    assert current[0]["attempt"] == 2
    assert current[0]["event_id"] != all_events[0]["event_id"]


@pytest.mark.asyncio
async def test_event_round_trip_preserves_neutral_continuation_metadata() -> None:
    store = ExecutionStore()
    await store.create("run-3", "start", {"run_id": "run-3"}, {"request": {"run_id": "run-3"}})

    continuation = {
        "binding_type": "langgraph_checkpoint",
        "payload": {"checkpoint_id": "checkpoint-1"},
        "binding_version": 2,
        "runtime_version": "runtime-7",
    }
    await store.append(
        "run-3",
        {
            "event_id": "run-3:paused",
            "kind": "run.interrupted",
            "payload": {"reason": "human_input"},
            "occurred_at": "2026-08-17T12:00:00Z",
            "trace_id": "trace-3",
            "runtime_version": "runtime-7",
            "contract_version": 1,
            "continuation": continuation,
            "terminal": True,
        },
    )

    events = await store.events_after("run-3")
    record = await store.get("run-3")
    assert events[0]["continuation"] == continuation
    assert events[0]["trace_id"] == "trace-3"
    assert events[0]["runtime_version"] == "runtime-7"
    assert events[0]["contract_version"] == 1
    assert record.continuation == continuation


@pytest.mark.asyncio
async def test_postgres_event_round_trip_updates_execution_continuation() -> None:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL runtime-store coverage")

    store = ExecutionStore(database_url.replace("postgresql+asyncpg://", "postgresql://", 1))
    await store.initialize()
    run_id = f"runtime-store-{uuid.uuid4().hex}"
    try:
        await store.create(run_id, "start", {"run_id": run_id}, {"request": {"run_id": run_id}})
        continuation = {
            "binding_type": "langgraph_checkpoint",
            "payload": {"checkpoint_id": "postgres-checkpoint"},
            "binding_version": 1,
            "runtime_version": "runtime-pg",
        }
        await store.append(
            run_id,
            {
                "event_id": f"{run_id}:paused",
                "kind": "run.interrupted",
                "payload": {"reason": "human_input"},
                "trace_id": "trace-pg",
                "runtime_version": "runtime-pg",
                "contract_version": 1,
                "continuation": continuation,
                "terminal": True,
            },
        )

        events = await store.events_after(run_id)
        record = await store.get(run_id)
        assert events[0]["continuation"] == continuation
        assert events[0]["trace_id"] == "trace-pg"
        assert record is not None
        assert record.continuation == continuation
    finally:
        # The Docker test runner drops the isolated database.  Keep direct
        # invocations tidy when they reuse a development test database.
        if store._pool is not None:
            await store._pool.execute("delete from runtime_executions where run_id=$1", run_id)
        await store.close()


def test_json_safe_converts_legacy_runtime_objects() -> None:
    value = _json_safe(
        {
            "run": SimpleNamespace(id="run-1"),
            "items": [SimpleNamespace(value=1)],
        }
    )

    assert value == {"run": {"id": "run-1"}, "items": [{"value": 1}]}
