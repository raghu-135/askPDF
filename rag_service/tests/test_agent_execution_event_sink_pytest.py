from __future__ import annotations

import asyncio

import pytest

from app.agent_workflows.execution_stream import (
    AgentExecutionEventSink,
    drain_retained_executions,
    retain_background_task,
)
from app.runtime.events import create_runtime_event


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

    assert [event.event_id for event in persisted] == ["run-1:1"]


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
