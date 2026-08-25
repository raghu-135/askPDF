from types import SimpleNamespace

from app.agent_workflows.canonical_trace import build_canonical_trace_projection
from app.agent_workflows.trace_recorder import AgentTraceRecorder
from app.runtime.contracts import AgentRuntimeEvent
from app.runtime.langgraph_adapter import _event_from_graph


def _event(sequence: int, kind: str, payload: dict, framework: str = "langgraph") -> AgentRuntimeEvent:
    return AgentRuntimeEvent(
        event_id=f"event-{sequence}",
        run_id="run-1",
        sequence=sequence,
        kind=kind,
        payload=payload,
        source_metadata={"framework": framework},
    )


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


def test_canonical_projection_never_synthesizes_an_operation_identity() -> None:
    projection = build_canonical_trace_projection(
        events=[_event(1, "operation.completed", {"operation_type": "unknown"}, "future")],
        resolved_spec={},
        framework="future",
    )

    assert projection["operations"] == []
    assert set(projection["visualizations"]) == {"generic.timeline"}
    assert projection["events"][0]["kind"] == "operation.completed"


def test_unknown_framework_metadata_remains_visible_without_specialized_visualization() -> None:
    projection = build_canonical_trace_projection(
        events=[_event(1, "runtime.event", {
            "message": "framework progress",
            "framework_details": {"future": {"phase": "inspect"}},
        }, "future")],
        resolved_spec={"config": {"graph": {"nodes": [{"id": "not-a-future-graph"}]}}},
        framework="future",
    )

    assert set(projection["visualizations"]) == {"generic.timeline"}
    assert projection["events"][0]["framework_details"]["future"]["phase"] == "inspect"


def test_tool_projection_records_argument_names_without_values() -> None:
    projection = build_canonical_trace_projection(
        events=[_event(1, "tool.completed", {
            "tool_name": "search_documents",
            "arguments": {"query": "private query", "authorization": "secret"},
            "ok": True,
        })],
        resolved_spec={},
        framework="langgraph",
    )

    payload = projection["tools"][0]["payload"]
    assert payload["provided_argument_names"] == ["authorization", "query"]
    assert "arguments" not in payload
    assert "private query" not in str(projection)


def test_hermes_projection_keeps_generic_events_and_session_visualization() -> None:
    events = [
        _event(1, "operation.started", {"operation_id": "hermes_session", "operation_type": "agent_session", "operation_label": "Hermes Agent", "session_id": "session-1"}, "hermes"),
        _event(2, "reasoning.available", {"session_id": "session-1", "upstream_run_id": "upstream-1"}, "hermes"),
        _event(3, "subagent.started", {"subagent_id": "child-1", "parent_subagent_id": "root"}, "hermes"),
        _event(4, "operation.completed", {"operation_id": "hermes_session", "operation_type": "agent_session", "operation_label": "Hermes Agent", "session_id": "session-1"}, "hermes"),
    ]

    projection = build_canonical_trace_projection(events=events, resolved_spec={}, framework="hermes")

    assert projection["operations"][0]["operation_label"] == "Hermes Agent"
    assert len(projection["events"]) == 4
    hermes = projection["visualizations"]["hermes.session"]
    assert hermes["session_id"] == "session-1"
    assert hermes["upstream_run_id"] == "upstream-1"
    assert len(hermes["reasoning"]) == 1
    assert len(hermes["subagents"]) == 1


def test_parallel_failures_remain_distinct_and_terminal_failure_correlates_them() -> None:
    events = [
        _event(1, "tool.failed", {
            "operation_id": "research-a",
            "tool_name": "search_documents",
            "error": {"code": "document_search_failed", "message": "Index unavailable"},
        }, "hermes"),
        _event(2, "subagent.failed", {
            "operation_id": "research-b",
            "subagent_id": "delegate-b",
            "error": {"code": "delegate_timeout", "message": "Delegate timed out"},
        }, "hermes"),
        _event(3, "run.failed", {
            "status": "failed",
            "error": {"code": "required_evidence_unavailable", "message": "Evidence unavailable"},
        }, "hermes"),
    ]

    projection = build_canonical_trace_projection(events=events, resolved_spec={}, framework="hermes")

    assert [row["event_id"] for row in projection["failures"]] == ["event-1", "event-2", "event-3"]
    assert projection["failures"][0]["classification"] == "primary"
    assert projection["failures"][1]["classification"] == "contributing"
    terminal = projection["failures"][-1]
    assert terminal["classification"] == "terminal"
    assert terminal["failure_count"] == 3
    assert terminal["primary_failure_event_id"] == "event-1"
    assert terminal["contributing_failure_event_ids"] == ["event-1", "event-2"]
    assert [row["event_id"] for row in projection["visualizations"]["hermes.session"]["failures"]] == [
        "event-1", "event-2", "event-3",
    ]


def test_trace_recorder_emits_version_two_from_canonical_events() -> None:
    run = SimpleNamespace(
        id="run-1",
        thread_id="thread-1",
        workflow_id="workflow-1",
        framework="future",
        status="completed",
        started_at=None,
        completed_at=None,
        resolved_spec_json={},
    )
    recorder = AgentTraceRecorder(run)
    recorder.record_agent_runtime_event(_event(1, "operation.completed", {
        "operation_id": "step-1",
        "operation_type": "agent_step",
        "operation_label": "Inspect",
        "visit_index": 1,
    }, "future"))

    payload = recorder.finalize(run=run, chat_turn_id=None, metrics={})

    assert payload["version"] == 2
    assert payload["operations"][0]["operation_id"] == "step-1"
    assert payload["trace"]["events"] == payload["events"]
    assert "graph" not in payload
