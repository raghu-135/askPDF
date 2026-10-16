import asyncio

import pytest

from langgraph_runtime.graph import NodeRegistry
from langgraph_runtime.workflows.enums import WorkflowNodeType
from langgraph_runtime.workflows.parallel_contracts import ParallelEventName


class _BlockingEmitSink:
    """Simulates a sink whose awaited emit would block worker completion."""

    def __init__(self) -> None:
        self.emit_nowait_calls: list[tuple[str, dict]] = []
        self._emit_entered = asyncio.Event()

    async def emit(self, kind: str, payload: dict) -> None:
        self._emit_entered.set()
        await asyncio.sleep(3600)

    def emit_nowait(self, kind: str, payload: dict) -> None:
        self.emit_nowait_calls.append((kind, payload))


@pytest.mark.asyncio
async def test_parallel_worker_lifecycle_emit_does_not_block_on_sink_ack(monkeypatch):
    registry = NodeRegistry()
    sink = _BlockingEmitSink()

    async def fast_worker(*_args, **_kwargs):
        return {
            "node_events": [{"status": "completed"}],
            "evidence_packets": [],
            "document_sources": [],
            "web_sources": [],
            "tool_events": [],
        }

    monkeypatch.setattr(registry, "_run_sequential_tool_worker", fast_worker)

    state = {
        "agent_run_id": "run-parallel-emit",
        "dispatch_node_id": WorkflowNodeType.PARALLEL_DISPATCH.value,
        "question": "What is the BLEU score?",
        "work_item": {
            "work_id": "work-1",
            "dispatch_id": "dispatch-1",
            "worker_node_id": WorkflowNodeType.RETRIEVAL_WORKER.value,
            "worker_type": WorkflowNodeType.RETRIEVAL_WORKER.value,
            "query": "BLEU score",
            "timeout_ms": 30_000,
            "max_attempts": 1,
            "ordinal": 0,
        },
    }
    config = {"configurable": {"execution_event_sink": sink}}

    result = await asyncio.wait_for(
        registry._parallel_tool_worker(WorkflowNodeType.RETRIEVAL_WORKER.value, state, config),
        timeout=2.0,
    )

    assert result["worker_result_packets"][0]["status"] == "completed"
    assert not sink._emit_entered.is_set()
    completed = [
        kind for kind, _payload in sink.emit_nowait_calls
        if kind == ParallelEventName.WORKER_COMPLETED
    ]
    assert completed
