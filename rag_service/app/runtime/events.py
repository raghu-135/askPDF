"""Construction and validation for canonical runtime events."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping

from app.agent_workflows.trace_sanitization import _bounded_value
from app.runtime.contracts import (
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
    value = str(kind or "").strip()
    source = dict(source_metadata or {})
    node_mapping = {
        "node.started": "operation.started",
        "node.completed": "operation.completed",
        "node.skipped": "operation.skipped",
        "node.failed": "operation.failed",
        "interrupt.created": "interrupt.requested",
        "run.interrupted": "interrupt.requested",
    }
    if value in {"model.started", "model.completed", "model.failed", "llm.start", "llm.complete"}:
        source.setdefault("source_event", value)
        value = {
            "model.started": "llm.started",
            "model.completed": "llm.completed",
            "model.failed": "llm.failed",
            "llm.start": "llm.started",
            "llm.complete": "llm.completed",
        }[value]
    if value in node_mapping:
        source.setdefault("source_event", value)
        value = node_mapping[value]
    if value not in CANONICAL_RUNTIME_EVENT_KINDS:
        source.setdefault("source_event", value or None)
        return "runtime.event", source
    return value, source


def normalize_product_event_kind(kind: str, *, source_metadata: Mapping[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Normalize product task events without making source metadata semantic."""

    value = str(kind or "").strip()
    source = dict(source_metadata or {})
    mapping = {
        "task.created": "run.queued",
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
    }
    if value.startswith("task.") and value.endswith("_requested"):
        action = value.removeprefix("task.").removesuffix("_requested")
        mapping[value] = {
            "cancel": "run.cancel_requested",
            "pause": "run.paused",
            "resume": "run.resumed",
        }.get(action, "runtime.event")
    if value.startswith("subagent."):
        status = value.removeprefix("subagent.")
        mapping[value] = {
            "start": "subagent.started",
            "started": "subagent.started",
            "progress": "subagent.progress",
            "complete": "subagent.completed",
            "completed": "subagent.completed",
            "failed": "subagent.failed",
            "timed_out": "subagent.failed",
            "cancelled": "subagent.cancelled",
        }.get(status, "runtime.event")
    if value.startswith("web_access."):
        mapping[value] = "approval.requested" if value.endswith("requested") else "approval.responded"
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
) -> AgentRuntimeEvent:
    if not str(event_id or "").strip():
        raise ValueError("runtime event_id is required")
    if not str(run_id or "").strip():
        raise ValueError("runtime run_id is required")
    if int(sequence) < 1:
        raise ValueError("runtime event sequence must be positive")
    if int(attempt) < 1:
        raise ValueError("runtime event attempt must be positive")
    normalized_kind, normalized_source = _event_kind(kind, source_metadata=source_metadata)
    expected_terminal = normalized_kind in TERMINAL_RUNTIME_EVENT_KINDS
    if terminal is not None and bool(terminal) != expected_terminal:
        raise ValueError(f"terminal flag does not match event kind {normalized_kind}")
    return AgentRuntimeEvent(
        event_id=str(event_id),
        run_id=str(run_id),
        sequence=int(sequence),
        kind=normalized_kind,
        attempt=int(attempt),
        payload=_bounded_value(dict(payload or {})),
        occurred_at=occurred_at or _now(),
        terminal=expected_terminal,
        trace_id=trace_id,
        source_metadata=_bounded_value(normalized_source),
        continuation=continuation,
    )


def validate_runtime_event(event: AgentRuntimeEvent, *, previous: AgentRuntimeEvent | None = None) -> None:
    if not event.event_id.strip():
        raise ValueError("runtime event_id is required")
    if not event.run_id.strip():
        raise ValueError("runtime run_id is required")
    if event.sequence < 1:
        raise ValueError("runtime event sequence must be positive")
    if event.kind not in CANONICAL_RUNTIME_EVENT_KINDS:
        raise ValueError(f"unsupported runtime event kind: {event.kind}")
    if event.terminal != (event.kind in TERMINAL_RUNTIME_EVENT_KINDS):
        raise ValueError(f"terminal flag does not match event kind {event.kind}")
    payload = dict(event.payload or {})
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
        if event.sequence <= previous.sequence:
            raise ValueError("runtime event sequence must be monotonic")
        if previous.terminal:
            raise ValueError("runtime events cannot follow a terminal event")
