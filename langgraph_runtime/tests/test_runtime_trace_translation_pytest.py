import time
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from langgraph_runtime.adapter import _event_from_graph, _LangGraphEventBridge
from langgraph_runtime.workflows.runtime_invocation import invoke_llm_for_node


@pytest.mark.asyncio
async def test_event_bridge_reports_completed_background_validation_failure():
    bridge = _LangGraphEventBridge("run", SimpleNamespace(emit_runtime_event=AsyncMock()))
    bridge.emit_nowait("tool.completed", {"tool_name": "search_documents"})
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    with pytest.raises(ValueError, match="tool_call_id"):
        await bridge.drain()


@pytest.mark.parametrize(("kind", "payload"), [
    ("output.delta", {"delta": "text"}),
    ("tool.started", {"tool_name": "search_documents", "tool_call_id": "call"}),
    ("subagent.started", {"subagent_id": "child"}),
    ("artifact.created", {"artifact_id": "artifact"}),
    ("approval.requested", {"approval_id": "approval", "response_operation": "run.approval.respond"}),
])
def test_translator_enforces_canonical_event_payloads(kind, payload):
    event = _event_from_graph({"event": kind, "data": payload}, run_id="run", sequence=1)
    assert event.kind == kind
    with pytest.raises(ValueError):
        _event_from_graph({"event": kind, "data": {}}, run_id="run", sequence=1)


@pytest.mark.parametrize("event", [{}, {"kind": "run.started"}, {"event": "unknown.event"}, {"event": "run.started", "data": []}])
def test_translator_rejects_unknown_or_malformed_upstream_envelopes(event):
    with pytest.raises(ValueError):
        _event_from_graph(event, run_id="run", sequence=1)


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


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("planner.repair_started", "operation.started"),
        ("planner.validation_failed", "operation.failed"),
        ("planner.failed", "operation.failed"),
    ],
)
def test_planner_validation_events_are_translated_to_canonical_operations(source, expected):
    from langgraph_runtime.runtime_support.observability import normalize_runtime_event

    kind, payload = normalize_runtime_event(source, {"category": "json_parse_error"})

    assert kind == expected
    assert payload["operation_id"] == "deep_task_planner"
    assert payload["operation_type"] == "deep_task_planner"
    assert payload["source_event"] == source
