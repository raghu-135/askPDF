from __future__ import annotations

import pytest

from runtime_protocol.contracts import (
    RuntimeEventKind,
    TERMINAL_RUNTIME_EVENT_KINDS,
)
from runtime_protocol.events import (
    create_runtime_event,
    normalize_product_event_kind,
    validate_runtime_event,
)


def test_canonical_event_factory_serializes_required_metadata():
    event = create_runtime_event(
        event_id="run-1:event-1",
        run_id="run-1",
        sequence=1,
        kind=RuntimeEventKind.TOOL_COMPLETED.value,
        payload={"tool_name": "search_documents"},
        source_metadata={"framework": "langgraph", "source_event": "tool.completed"},
    )

    value = event.to_dict()
    assert value["kind"] == "tool.completed"
    assert value["run_id"] == "run-1"
    assert value["sequence"] == 1
    assert value["occurred_at"]
    assert value["terminal"] is False
    assert value["source_metadata"]["framework"] == "langgraph"


def test_factory_rejects_invalid_terminal_and_sequence_values():
    with pytest.raises(ValueError, match="sequence must be positive"):
        create_runtime_event(event_id="event", run_id="run", sequence=0, kind="run.started")
    with pytest.raises(ValueError, match="terminal flag"):
        create_runtime_event(
            event_id="event",
            run_id="run",
            sequence=1,
            kind="run.completed",
            terminal=False,
        )


def test_terminal_events_are_unique_and_cannot_be_followed():
    terminal = create_runtime_event(
        event_id="run:terminal",
        run_id="run",
        sequence=2,
        kind="run.completed",
    )
    assert terminal.kind in TERMINAL_RUNTIME_EVENT_KINDS
    with pytest.raises(ValueError, match="follow a terminal"):
        validate_runtime_event(
            create_runtime_event(event_id="run:after", run_id="run", sequence=3, kind="run.failed"),
            previous=terminal,
        )


def test_unknown_source_event_is_observational_runtime_event():
    event = create_runtime_event(
        event_id="run:unknown",
        run_id="run",
        sequence=1,
        kind="hermes.future.event",
        payload={"value": 1},
    )
    assert event.kind == "runtime.event"
    assert event.source_metadata["source_event"] == "hermes.future.event"


def test_clarification_is_a_canonical_terminal_event():
    event = create_runtime_event(
        event_id="run:clarification",
        run_id="run",
        sequence=1,
        kind=RuntimeEventKind.RUN_CLARIFICATION.value,
    )
    assert event.kind == "run.clarification"
    assert event.terminal is True


@pytest.mark.parametrize("kind", ["dispatch.started", "worker.retrying", "worker.timed_out", "aggregation.partial"])
def test_parallel_lifecycle_events_remain_canonical(kind):
    event = create_runtime_event(event_id=f"run:{kind}", run_id="run", sequence=1, kind=kind)
    assert event.kind == kind
    assert event.source_metadata == {}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("task.completed", "run.completed"),
        ("task.cancel_requested", "run.cancel_requested"),
        ("subagent.start", "subagent.started"),
        ("subagent.complete", "subagent.completed"),
        ("artifact.invalidated", "artifact.updated"),
    ],
)
def test_product_event_sources_normalize_to_neutral_kinds(source: str, expected: str):
    kind, metadata = normalize_product_event_kind(source)
    assert kind == expected
    assert metadata["source_event"] == source
