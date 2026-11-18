from __future__ import annotations

import pytest

from app.product_orchestration.parallel_event_reconcile import reconcile_parallel_terminal_events


class _Sink:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []
        self._events: list = []

    def canonical_events(self):
        return self._events

    async def emit(self, kind: str, payload: dict) -> None:
        self.emitted.append((kind, payload))


@pytest.mark.asyncio
async def test_reconcile_emits_missing_worker_and_aggregation_terminal_events():
    sink = _Sink()

    class _Event:
        def __init__(self, kind: str, payload: dict):
            self.kind = kind
            self.payload = payload

    sink._events = [
        _Event("dispatch.started", {"dispatch_id": "dispatch-1"}),
        _Event("worker.started", {"dispatch_id": "dispatch-1", "work_id": "work-web"}),
    ]
    result = {
        "agent_run_id": "run-1",
        "dispatch_id": "dispatch-1",
        "parallel_summary": {
            "dispatch_id": "dispatch-1",
            "parent_operation_id": "parallel_dispatch",
            "barrier_state": "reached",
            "aggregation_state": "completed",
            "planned": 1,
            "completed": 1,
        },
        "worker_result_packets": [
            {
                "work_id": "work-web",
                "worker_node_id": "web_worker",
                "worker_type": "web_worker",
                "status": "completed",
                "attempt": 1,
            }
        ],
    }
    await reconcile_parallel_terminal_events(sink, result)
    kinds = [kind for kind, _payload in sink.emitted]
    assert "worker.completed" in kinds
    assert "dispatch.barrier_reached" in kinds
    assert "aggregation.completed" in kinds