from __future__ import annotations

import pytest

from app.product_orchestration.execution_stream import AgentExecutionEventSink


@pytest.mark.asyncio
async def test_live_delivery_includes_canonical_run_id_on_operation_events() -> None:
    sink = AgentExecutionEventSink(include_details=False)

    async def persist(*_args, **_kwargs):
        return None

    sink.bind_runtime_event_persister("run-abc", persist)
    await sink.emit("operation.started", {"operation_id": "planner", "visit_index": 1})
    item = sink.queue.get_nowait()
    assert item["event"] == "operation.started"
    assert item["data"]["run_id"] == "run-abc"
    assert item["data"]["operation_id"] == "planner"
    await sink.finish_boundary()


@pytest.mark.asyncio
async def test_terminal_event_preserves_long_answer_on_live_delivery() -> None:
    sink = AgentExecutionEventSink(include_details=False)
    long_answer = "a" * 1200

    async def persist(*_args, **_kwargs):
        return None

    sink.bind_runtime_event_persister("run-abc", persist)
    await sink.finish(
        "run.completed",
        {
            "run_id": "run-abc",
            "status": "completed",
            "response": {"answer": long_answer, "final_answer": long_answer},
        },
    )
    item = sink.queue.get_nowait()
    assert item["event"] == "run.completed"
    assert item["data"]["response"]["answer"] == long_answer
    assert item["data"]["response"]["final_answer"] == long_answer
    assert len(item["data"]["response"]["answer"]) > 900
