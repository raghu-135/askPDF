from __future__ import annotations

import asyncio

import pytest

from app.agent_workflows.execution_stream import (
    AgentExecutionEventSink,
    drain_retained_executions,
    retain_background_task,
)
from app.runtime.events import RuntimeEventContractViolation, create_runtime_event


@pytest.mark.asyncio
async def test_event_writer_persists_fifo_before_live_delivery():
    persisted = []
    release = asyncio.Event()
    sink = AgentExecutionEventSink()

    async def persist(_run_id, event):
        if event.kind == "tool.started":
            await release.wait()
        persisted.append((event.sequence, event.kind))

    sink.bind_runtime_event_persister("run-1", persist)
    sink.emit_nowait("tool.started", {"event_id": "tool-1"})
    sink.emit_nowait("tool.completed", {"event_id": "tool-2"})
    await asyncio.sleep(0)
    assert sink.queue.empty()

    release.set()
    await sink.flush()
    assert persisted == [(1, "tool.started"), (2, "tool.completed")]
    assert [await sink.queue.get(), await sink.queue.get()] == [
        {"event": "tool.started", "data": {"event_id": "tool-1"}},
        {"event": "tool.completed", "data": {"event_id": "tool-2"}},
    ]
    await sink.finish("run.completed", {"status": "completed"})


@pytest.mark.asyncio
async def test_live_delivery_includes_backend_projected_parallel_groups():
    sink = AgentExecutionEventSink()
    sink.bind_runtime_event_persister("run-1", lambda _run_id, event: _append([], event))

    await sink.emit("dispatch.started", {"event_id": "dispatch-event", "dispatch_id": "dispatch-1", "planned": 1})
    await sink.emit("worker.started", {
        "event_id": "worker-event",
        "dispatch_id": "dispatch-1",
        "work_id": "work-1",
        "operation_id": "retrieval-worker",
        "ordinal": 0,
        "attempt": 1,
    })

    dispatch_frame = await sink.queue.get()
    worker_frame = await sink.queue.get()
    assert dispatch_frame["data"]["parallel_groups"][0]["group_id"] == "dispatch-1"
    group = worker_frame["data"]["parallel_groups"][0]
    assert group["members"][0]["member_id"] == "work-1"
    assert group["members"][0]["operation_id"] == "retrieval-worker"
    await sink.finish("run.completed", {"status": "completed"})


@pytest.mark.asyncio
async def test_event_writer_fails_fast_for_malformed_parallel_identity():
    sink = AgentExecutionEventSink()
    sink.bind_runtime_event_persister("run-contract", lambda _run_id, event: _append([], event))

    with pytest.raises(RuntimeEventContractViolation, match="group identity") as error:
        await sink.emit("worker.started", {"work_id": "work-1", "attempt": 1})

    assert error.value.code == "debug_trace_contract_violation"
    assert error.value.retryable is False
    assert error.value.correlation_id == "trace:run-contract"


@pytest.mark.asyncio
async def test_event_writer_continues_persisted_sequence_and_rejects_after_terminal():
    persisted = []
    sink = AgentExecutionEventSink()

    async def persist(_run_id, event):
        persisted.append(event)

    sink.bind_runtime_event_persister("run-1", persist, initial_sequence=7)
    await sink.emit("run.started", {"resumed": True})
    await sink.finish("run.completed", {"status": "completed"})

    assert [(event.sequence, event.kind) for event in persisted] == [
        (8, "run.started"),
        (9, "run.completed"),
    ]
    with pytest.raises(RuntimeError, match="finalized"):
        await sink.emit("tool.completed", {})
    assert len(persisted) == 2


@pytest.mark.asyncio
async def test_event_writer_is_idempotent_but_rejects_conflicting_duplicate_ids():
    persisted = []
    sink = AgentExecutionEventSink()

    async def persist(_run_id, event):
        persisted.append(event)

    sink.bind_runtime_event_persister("run-1", persist)
    await sink.emit("tool.progress", {"event_id": "same", "value": 1})
    await sink.emit("tool.progress", {"event_id": "same", "value": 1})
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        await sink.emit("tool.progress", {"event_id": "same", "value": 2})
    assert len(persisted) == 1
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        await sink.finish_boundary()


@pytest.mark.asyncio
async def test_detach_delivery_preserves_persistence():
    persisted = []
    sink = AgentExecutionEventSink()
    sink.bind_runtime_event_persister("run-1", lambda _run_id, event: _append(persisted, event))
    sink.detach_delivery()

    await sink.emit("tool.completed", {})
    await sink.finish("run.completed", {"status": "completed"})

    assert [event.kind for event in persisted] == ["tool.completed", "run.completed"]
    assert sink.queue.empty()


async def _append(values, value):
    values.append(value)


@pytest.mark.asyncio
async def test_transport_terminal_is_not_recorded_as_product_terminal():
    persisted = []
    sink = AgentExecutionEventSink()
    sink.bind_runtime_event_persister("run-1", lambda _run_id, event: _append(persisted, event))
    runtime_terminal = create_runtime_event(
        event_id="runtime-terminal",
        run_id="run-1",
        sequence=1,
        kind="run.completed",
    )

    await sink.emit_runtime_event(runtime_terminal)
    await sink.finish("run.completed", {"status": "completed"})

    assert [event.event_id for event in persisted] == ["askpdf-terminal:run-1:run.completed"]


@pytest.mark.asyncio
async def test_product_terminal_id_cannot_collide_with_runtime_sequence_id():
    persisted = []
    sink = AgentExecutionEventSink()
    sink.bind_runtime_event_persister("run-1", lambda _run_id, event: _append(persisted, event))

    await sink.emit("output.completed", {"event_id": "run-1:2"})
    await sink.finish("run.completed", {"status": "completed"})

    assert [event.event_id for event in persisted] == [
        "run-1:2",
        "askpdf-terminal:run-1:run.completed",
    ]


@pytest.mark.asyncio
async def test_writer_failure_is_observed_and_failure_terminal_can_still_be_persisted():
    persisted = []
    sink = AgentExecutionEventSink()

    async def persist(_run_id, event):
        if event.kind == "tool.progress":
            raise RuntimeError("event persistence unavailable")
        persisted.append(event)

    sink.bind_runtime_event_persister("run-1", persist)
    sink.emit_nowait("tool.progress", {"event_id": "failed-write"})

    with pytest.raises(RuntimeError, match="event persistence unavailable"):
        await sink.flush()
    await sink.finish("run.failed", {"status": "failed"})

    assert [(event.sequence, event.kind) for event in persisted] == [(2, "run.failed")]
    assert [await sink.queue.get()] == [{"event": "run.failed", "data": {"status": "failed"}}]


@pytest.mark.asyncio
async def test_transactional_terminal_is_delivered_only_after_commit_acknowledgement():
    sink = AgentExecutionEventSink()
    sink.bind_runtime_event_persister("run-1", lambda _run_id, event: _append([], event))
    commit_started = asyncio.Event()
    release_commit = asyncio.Event()

    async def commit(_event):
        commit_started.set()
        await release_commit.wait()

    finish = asyncio.create_task(
        sink.finish("run.completed", {"status": "completed"}, terminal_committer=commit)
    )
    await commit_started.wait()
    assert sink.queue.empty()
    release_commit.set()
    await finish
    assert await sink.queue.get() == {"event": "run.completed", "data": {"status": "completed"}}


@pytest.mark.asyncio
async def test_transactional_terminal_is_in_recorder_before_debug_commit():
    sink = AgentExecutionEventSink()
    sink.bind_runtime_event_persister("run-1", lambda _run_id, event: _append([], event))

    class Recorder:
        def __init__(self):
            self.events = []

        def record_agent_runtime_event(self, event):
            self.events.append(event)

    recorder = Recorder()
    sink.bind_trace_recorder(recorder)
    observed = []

    async def commit(terminal_event):
        observed.append(([event.kind for event in recorder.events], terminal_event.payload))

    await sink.emit("tool.failed", {"error": {"code": "first_failure"}})
    await sink.emit("subagent.failed", {"error": {"code": "second_failure"}})
    await sink.finish(
        "run.failed",
        {"status": "failed", "error": {"code": "terminal_failure"}},
        terminal_committer=commit,
    )

    assert observed == [(
        ["tool.failed", "subagent.failed", "run.failed"],
        {"status": "failed", "error": {"code": "terminal_failure"}},
    )]


@pytest.mark.asyncio
async def test_transactional_terminal_commit_failure_emits_no_terminal_frame():
    sink = AgentExecutionEventSink()
    sink.bind_runtime_event_persister("run-1", lambda _run_id, event: _append([], event))

    async def fail(_event):
        raise RuntimeError("terminal commit failed")

    with pytest.raises(RuntimeError, match="terminal commit failed"):
        await sink.finish("run.completed", {"status": "completed"}, terminal_committer=fail)
    assert sink.queue.empty()


@pytest.mark.asyncio
async def test_retained_execution_shutdown_cancels_overdue_tasks():
    cancelled = asyncio.Event()

    async def run_forever():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    task = asyncio.create_task(run_forever())
    retain_background_task(task)
    await drain_retained_executions(0)

    assert task.cancelled()
    assert cancelled.is_set()
