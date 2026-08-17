from __future__ import annotations

import os
import uuid
from types import SimpleNamespace

import pytest

from runtime_service.execution_store import ExecutionConflictError, ExecutionStore, LeaseLostError, _json_safe


@pytest.mark.asyncio
async def test_terminal_continuation_probe_is_immutable_under_repeated_start() -> None:
    store = ExecutionStore()

    await store.create("run-1", "continue_run", {"run_id": "run-1"}, {"request": {"run_id": "run-1"}})
    await store.set_status("run-1", "no_continuation")
    await store.append(
        "run-1",
        {"event_id": "run-1:terminal", "kind": "run.continuation_empty", "terminal": True, "payload": {}},
    )

    with pytest.raises(ExecutionConflictError):
        await store.create("run-1", "start", {"run_id": "run-1"}, {"request": {"run_id": "run-1"}})
    assert (await store.get("run-1")).status == "no_continuation"
    assert len(await store.events_after("run-1")) == 1


@pytest.mark.asyncio
async def test_failed_start_is_immutable_under_transport_retry() -> None:
    store = ExecutionStore()

    await store.create("run-2", "start", {"run_id": "run-2"}, {"request": {"run_id": "run-2"}})
    await store.set_status("run-2", "failed")
    await store.append(
        "run-2",
        {"event_id": "run-2:terminal", "kind": "run.failed", "terminal": True, "payload": {}},
    )

    record = await store.create("run-2", "start", {"run_id": "run-2"}, {"request": {"run_id": "run-2"}})

    assert record.status == "failed"
    assert len(await store.events_after("run-2")) == 1


@pytest.mark.asyncio
async def test_terminal_request_conflict_requires_explicit_retry() -> None:
    store = ExecutionStore()
    await store.create("run-retry", "start", {"run_id": "run-retry", "input": {"question": "one"}}, {"request": {"run_id": "run-retry"}})
    await store.set_status("run-retry", "cancelled")

    with pytest.raises(ExecutionConflictError):
        await store.create("run-retry", "start", {"run_id": "run-retry", "input": {"question": "two"}}, {"request": {"run_id": "run-retry"}})

    retried = await store.create(
        "run-retry",
        "retry",
        {"run_id": "run-retry", "retry_operation": "start", "retry_request": {"run_id": "run-retry"}},
        {"request": {"run_id": "run-retry"}},
        operation_id="retry-1",
        source_attempt=1,
    )
    assert retried.status == "queued"
    assert retried.attempt == 2
    repeated = await store.create(
        "run-retry",
        "retry",
        {"run_id": "run-retry", "retry_operation": "start", "retry_request": {"run_id": "run-retry"}},
        {"request": {"run_id": "run-retry"}},
        operation_id="retry-1",
        source_attempt=1,
    )
    assert repeated.attempt == 2


@pytest.mark.asyncio
async def test_resume_transport_retry_is_read_only_after_terminal_completion() -> None:
    store = ExecutionStore()
    await store.create("run-resume", "start", {"run_id": "run-resume"}, {"request": {"run_id": "run-resume"}})
    binding = {"binding_type": "checkpoint", "payload": {"id": "cp-1"}}
    await store.append(
        "run-resume",
        {"event_id": "run-resume:terminal", "kind": "run.completed", "terminal": True, "continuation": binding},
    )
    await store.set_status("run-resume", "completed")

    with pytest.raises(ExecutionConflictError):
        await store.create("run-resume", "resume", {"run_id": "run-resume"}, {"request": {"run_id": "run-resume"}})
    assert (await store.get("run-resume")).attempt == 1
    assert len(await store.events_after("run-resume")) == 1


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
    os.environ["AGENT_RUNTIME_SCHEMA_AUTO_CREATE"] = "true"
    await store.initialize()
    run_id = f"runtime-store-{uuid.uuid4().hex}"
    try:
        await store.create(run_id, "start", {"run_id": run_id}, {"request": {"run_id": run_id}})
        fencing_token = await store.claim(run_id)
        assert fencing_token is not None
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
            owner_id=store.owner_id,
            fencing_token=fencing_token,
        )
        with pytest.raises(LeaseLostError):
            await store.append(
                run_id,
                {"event_id": f"{run_id}:stale", "kind": "runtime.event"},
                owner_id="stale-worker",
                fencing_token=fencing_token,
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


@pytest.mark.asyncio
async def test_runtime_lease_fences_competing_workers_and_mutations() -> None:
    store = ExecutionStore()
    await store.create("leased", "start", {"run_id": "leased"}, {"request": {"run_id": "leased"}})
    first = await store.claim("leased", owner_id="worker-a")
    assert first is not None
    assert await store.claim("leased", owner_id="worker-b") is None
    with pytest.raises(LeaseLostError):
        await store.append("leased", {"event_id": "stale", "kind": "runtime.event"}, owner_id="worker-b", fencing_token=1)
    assert await store.heartbeat("leased", owner_id="worker-a", fencing_token=first)
