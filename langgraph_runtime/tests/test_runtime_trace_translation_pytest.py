import time
from types import SimpleNamespace

import pytest

from langgraph_runtime.adapter import _event_from_graph
from langgraph_runtime.workflows.runtime_invocation import invoke_llm_for_node


def test_langgraph_node_translation_preserves_operation_identity_and_topology() -> None:
    event = _event_from_graph(
        {
            "event": "node.completed",
            "data": {
                "node_id": "retrieval_1",
                "node_type": "retrieval_worker",
                "label": "Document retrieval",
                "visit_index": 2,
                "route": "answer",
                "duration_ms": 12,
            },
        },
        run_id="run-1",
        sequence=1,
    )

    assert event.kind == "operation.completed"
    assert event.payload["operation_id"] == "retrieval_1"
    assert event.payload["operation_type"] == "retrieval_worker"
    assert event.payload["operation_label"] == "Document retrieval"
    assert event.payload["visit_index"] == 2
    assert event.payload["topology_ref"] == {"kind": "graph_node", "id": "retrieval_1"}
    assert event.payload["framework_details"]["langgraph"]["route"] == "answer"


@pytest.mark.asyncio
async def test_shared_model_invocation_emits_bounded_lifecycle_events() -> None:
    class Sink:
        def __init__(self) -> None:
            self.events = []

        async def emit(self, kind, payload):
            self.events.append((kind, payload))

    sink = Sink()
    response = SimpleNamespace(content="safe result", usage_metadata={"total_tokens": 9})

    async def invoke(_messages):
        return response

    await invoke_llm_for_node(
        invoke,
        [],
        state={"llm_model": "test-model", "agent_run_id": "run-1"},
        config={"configurable": {"execution_event_sink": sink}},
        node="planner",
        started=time.perf_counter(),
        retry_observer=lambda _event: None,
        retry_attempts=[],
        model_name="test-model",
    )

    assert [kind for kind, _payload in sink.events] == ["llm.started", "llm.completed"]
    assert sink.events[0][1]["operation_id"] == "planner"
    assert sink.events[1][1]["usage"]["total_tokens"] == 9
    assert "messages" not in str(sink.events)
