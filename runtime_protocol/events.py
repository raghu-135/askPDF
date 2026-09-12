"""Construction and validation for canonical runtime events."""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
from typing import Any, Mapping

from runtime_protocol.sanitization import bounded_value
from runtime_protocol.contracts import (
    AgentRuntimeEvent,
    CANONICAL_RUNTIME_EVENT_KINDS,
    TERMINAL_RUNTIME_EVENT_KINDS,
)


class RuntimeEventContractViolation(ValueError):
    code = "debug_trace_contract_violation"
    retryable = False

    def __init__(self, message: str, *, field_path: str = "runtime_event", correlation_id: str | None = None) -> None:
        super().__init__(message)
        self.field_path = field_path
        self.correlation_id = correlation_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _event_kind(kind: str, *, source_metadata: Mapping[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    if not isinstance(kind, str) or kind not in CANONICAL_RUNTIME_EVENT_KINDS:
        raise ValueError(f"unsupported runtime event kind: {kind}")
    if source_metadata is not None and not isinstance(source_metadata, Mapping):
        raise ValueError("runtime event source_metadata must be an object")
    return kind, dict(source_metadata or {})


def canonical_event_payload(kind: str, payload: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """Accept only canonical kinds at the product boundary; adapters translate upstream events."""
    _event_kind(kind)
    if payload is not None and not isinstance(payload, Mapping):
        raise ValueError("runtime event payload must be an object")
    return kind, dict(bounded_value(dict(payload) if payload is not None else {}))


def normalize_product_event_kind(kind: str, *, source_metadata: Mapping[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Normalize product task events without making source metadata semantic."""

    value = str(kind or "").strip()
    source = dict(source_metadata or {})
    mapping = {
        "task.created": "run.queued",
        "task.queued": "run.queued",
        "task.pausing": "runtime.event",
        "task.awaiting_approval": "run.paused",
        "task.cancelling": "run.cancel_requested",
        "task.recovery_required": "runtime.event",
        "task.start_requested": "run.queued",
        "task.retry_requested": "run.queued",
        "task.approval_requested": "interrupt.requested",
        "task.result_review_requested": "interrupt.requested",
        "todo.started": "operation.started",
        "todo.running": "operation.started",
        "todo.completed": "operation.completed",
        "todo.failed": "operation.failed",
        "todo.skipped": "operation.skipped",
        "todo.cancelled": "operation.skipped",
        "todo.pending": "runtime.event",
        "todo.ready": "runtime.event",
        "todo.blocked": "runtime.event",
        "task.claimed": "run.started",
        "task.run_attached": "run.started",
        "task.continuation_queued": "run.queued",
        "task.approval_resolved": "approval.responded",
        "task.deletion_requested": "run.cancel_requested",
        "task.paused": "run.paused",
        "task.resumed": "run.resumed",
        "task.running": "run.started",
        "task.completed": "run.completed",
        "task.failed": "run.failed",
        "task.expired": "run.failed",
        "task.cancelled": "run.cancelled",
        "artifact.deleted": "artifact.updated",
        "artifact.invalidated": "artifact.updated",
        "web_access.allowed_for_task": "approval.responded",
        "web_access.denied_for_task": "approval.responded",
        "task.course_correction_submitted": "course_correction.accepted",
        "task.course_correction_incorporated": "course_correction.incorporated",
        "task.course_correction_accepted_unresolved": "course_correction.unresolved",
        "task.course_correction_rejected": "course_correction.unresolved",
        "task.course_correction_satisfied": "course_correction.satisfied",
        "task.course_correction_unresolved": "course_correction.unresolved",
        "task.course_correction_linked": "linked_run.created",
        "task.budget_review_requested": "budget.boundary_requested",
        "task.budget_review_continued": "intervention.responded",
        "task.budget_review_partial_accepted": "intervention.responded",
        "task.budget_review_steered": "intervention.responded",
        "task.budget_updated": "runtime.event",
        "task.deletion_completed": "runtime.event",
        "task.lease_recovered": "runtime.event",
        "task.result_review_accepted": "intervention.responded",
        "task.result_review_retry_queued": "run.queued",
        "task.runtime_projection_failed": "runtime.event",
        "task.runtime_projection_reconciled": "runtime.event",
        "task.wake_budget_reached": "budget.boundary_requested",
    }
    if value.startswith("task.") and value.endswith("_requested"):
        action = value.removeprefix("task.").removesuffix("_requested")
        requested_kind = {
            "cancel": "run.cancel_requested",
            "pause": "run.paused",
            "resume": "run.resumed",
        }.get(action)
        if requested_kind is not None:
            mapping[value] = requested_kind
    if value.startswith("subagent."):
        status = value.removeprefix("subagent.")
        subagent_kind = {
            "start": "subagent.started",
            "started": "subagent.started",
            "running": "subagent.started",
            "progress": "subagent.progress",
            "complete": "subagent.completed",
            "completed": "subagent.completed",
            "failed": "subagent.failed",
            "timed_out": "subagent.failed",
            "cancelled": "subagent.cancelled",
        }.get(status)
        if subagent_kind is not None:
            mapping[value] = subagent_kind
    normalized = mapping.get(value)
    if normalized is None:
        normalized, source = _event_kind(value, source_metadata=source)
    if normalized != value:
        source.setdefault("source_event", value)
    return normalized, source


def create_runtime_event(
    *,
    event_id: str,
    run_id: str,
    sequence: int,
    kind: str,
    payload: Mapping[str, Any] | None = None,
    attempt: int = 1,
    occurred_at: str | None = None,
    terminal: bool | None = None,
    trace_id: str | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    continuation: Any = None,
    checkpoint_boundary_available: bool | None = None,
) -> AgentRuntimeEvent:
    if not isinstance(event_id, str) or not event_id.strip():
        raise ValueError("runtime event_id is required")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("runtime run_id is required")
    if type(sequence) is not int or sequence < 1:
        raise ValueError("runtime event sequence must be positive")
    if type(attempt) is not int or attempt < 1:
        raise ValueError("runtime event attempt must be positive")
    if payload is not None and not isinstance(payload, Mapping):
        raise ValueError("runtime event payload must be an object")
    normalized_kind, normalized_source = _event_kind(kind, source_metadata=source_metadata)
    expected_terminal = normalized_kind in TERMINAL_RUNTIME_EVENT_KINDS
    if terminal is not None and (type(terminal) is not bool or terminal != expected_terminal):
        raise ValueError(f"terminal flag does not match event kind {normalized_kind}")
    event = AgentRuntimeEvent(
        event_id=str(event_id),
        run_id=str(run_id),
        sequence=int(sequence),
        kind=normalized_kind,
        attempt=int(attempt),
        payload=dict(payload) if payload is not None else {},
        occurred_at=occurred_at or _now(),
        terminal=expected_terminal,
        trace_id=trace_id,
        source_metadata=bounded_value(normalized_source),
        continuation=continuation,
        checkpoint_boundary_available=checkpoint_boundary_available,
    )
    validate_runtime_event(event)
    # Validate the original types before redaction can turn values into strings.
    event = replace(event, payload=bounded_value(dict(event.payload)))
    return event


def validate_runtime_event(event: AgentRuntimeEvent, *, previous: AgentRuntimeEvent | None = None) -> None:
    if not isinstance(event.event_id, str) or not event.event_id.strip():
        raise ValueError("runtime event_id is required")
    if not isinstance(event.run_id, str) or not event.run_id.strip():
        raise ValueError("runtime run_id is required")
    if type(event.sequence) is not int or event.sequence < 1:
        raise ValueError("runtime event sequence must be positive")
    if event.kind not in CANONICAL_RUNTIME_EVENT_KINDS:
        raise ValueError(f"unsupported runtime event kind: {event.kind}")
    if type(event.terminal) is not bool or event.terminal != (event.kind in TERMINAL_RUNTIME_EVENT_KINDS):
        raise ValueError(f"terminal flag does not match event kind {event.kind}")
    if type(event.attempt) is not int or event.attempt < 1:
        raise ValueError("runtime event attempt must be positive")
    if not isinstance(event.payload, Mapping):
        raise ValueError("runtime event payload must be an object")
    if not isinstance(event.source_metadata, Mapping):
        raise ValueError("runtime event source_metadata must be an object")
    payload = dict(event.payload)
    required = ()
    if event.kind.startswith("tool."):
        required = ("tool_call_id", "tool_name")
    elif event.kind.startswith("subagent."):
        required = ("subagent_id",)
    elif event.kind.startswith("artifact."):
        required = ("artifact_id",)
    elif event.kind.startswith("approval."):
        required = ("approval_id",)
    for name in required:
        if not isinstance(payload.get(name), str) or not payload[name].strip():
            raise ValueError(f"{event.kind} requires {name}")
    if event.kind == "output.delta" and not isinstance(payload.get("delta"), str):
        raise ValueError("output.delta requires a string delta")
    if event.kind == "approval.requested" and payload.get("response_operation") != "run.approval.respond":
        raise ValueError("approval.requested requires run.approval.respond")

    if payload.get("parent_operation_id") and payload.get("parent_operation_id") == payload.get("operation_id"):
        raise ValueError("runtime event operation cannot parent itself")
    caused_by = payload.get("caused_by_event_id")
    if caused_by is not None and (not isinstance(caused_by, str) or not caused_by.strip()):
        raise ValueError("runtime event caused_by_event_id must be a non-empty string")
    related = payload.get("related_event_ids")
    if related is not None and (
        not isinstance(related, list)
        or any(not isinstance(value, str) or not value.strip() for value in related)
    ):
        raise ValueError("runtime event related_event_ids must be an array of non-empty strings")
    if event.kind.startswith(("dispatch.", "worker.", "aggregation.")):
        group_id = payload.get("parallel_group_id", payload.get("dispatch_id", payload.get("wave_id")))
        if group_id is None or not str(group_id).strip():
            raise ValueError("parallel runtime event requires a group identity")
        mode = str(payload.get("dispatch_mode") or payload.get("mode") or "parallel").strip().lower()
        if mode not in {"serial", "parallel"}:
            raise ValueError("parallel runtime event dispatch_mode must be serial or parallel")
        if event.kind.startswith("worker."):
            member_id = payload.get("work_id") or payload.get("operation_id")
            if member_id is None or not str(member_id).strip():
                raise ValueError("worker runtime event requires a member identity")
            try:
                if int(payload.get("attempt") or event.attempt) < 1:
                    raise ValueError
            except (TypeError, ValueError) as exc:
                raise ValueError("worker runtime event attempt must be positive") from exc
    if previous is not None:
        if event.run_id != previous.run_id:
            raise ValueError("runtime events must belong to the same run")
        if event.sequence <= previous.sequence:
            raise ValueError("runtime event sequence must be monotonic")
        if previous.terminal:
            raise ValueError("runtime events cannot follow a terminal event")
