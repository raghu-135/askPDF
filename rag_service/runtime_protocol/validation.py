"""Strict validation for runtime result envelopes."""

from __future__ import annotations

from typing import Any, Mapping

from runtime_protocol.contracts import RuntimeEventKind, RuntimeTaskResultStatus


RUNTIME_RESULT_STATUSES = frozenset({
    "completed",
    "no_continuation",
    "clarification_required",
    "awaiting_human",
    "paused",
    "failed",
    "timed_out",
    "cancelled",
})

_EVENT_RESULT_STATUSES: dict[str, frozenset[str]] = {
    RuntimeEventKind.RUN_COMPLETED.value: frozenset({"completed", "no_continuation"}),
    RuntimeEventKind.RUN_CLARIFICATION.value: frozenset({"clarification_required"}),
    RuntimeEventKind.RUN_FAILED.value: frozenset({"failed", "timed_out"}),
    RuntimeEventKind.RUN_CANCELLED.value: frozenset({"cancelled"}),
    RuntimeEventKind.RUN_PAUSED.value: frozenset({"awaiting_human", "paused"}),
    RuntimeEventKind.INTERRUPT_REQUESTED.value: frozenset({"awaiting_human", "paused"}),
    RuntimeEventKind.APPROVAL_REQUESTED.value: frozenset({"awaiting_human", "paused"}),
}


class RuntimeProtocolValidationError(ValueError):
    """A runtime payload violates the framework-neutral wire contract."""

    code = "runtime_protocol_error"

    def __init__(self, message: str, *, field: str, value: Any = None) -> None:
        self.field = field
        self.value = value
        super().__init__(message)

    def details(self) -> dict[str, Any]:
        details: dict[str, Any] = {"field": self.field}
        if self.value is not None and isinstance(self.value, (str, int, float, bool)):
            details["received"] = self.value
        return details


def validate_runtime_result_envelope(value: Mapping[str, Any]) -> None:
    """Validate required result status fields without coercion or fallback."""

    if not isinstance(value, Mapping):
        raise RuntimeProtocolValidationError(
            "runtime result envelope must be an object", field="result"
        )
    status = value.get("status")
    if not isinstance(status, str) or not status.strip():
        raise RuntimeProtocolValidationError(
            "runtime result envelope requires a non-empty status", field="status", value=status
        )
    if status not in RUNTIME_RESULT_STATUSES:
        allowed = ", ".join(sorted(RUNTIME_RESULT_STATUSES))
        raise RuntimeProtocolValidationError(
            f"unknown runtime result status {status!r}; expected one of: {allowed}",
            field="status",
            value=status,
        )

    task_result = value.get("task_result")
    if task_result is None:
        return
    if not isinstance(task_result, Mapping):
        raise RuntimeProtocolValidationError(
            "task_result must be an object", field="task_result"
        )
    task_status = task_result.get("status")
    if not isinstance(task_status, str) or not task_status.strip():
        raise RuntimeProtocolValidationError(
            "task_result requires a non-empty status",
            field="task_result.status",
            value=task_status,
        )
    try:
        RuntimeTaskResultStatus(task_status)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in RuntimeTaskResultStatus)
        raise RuntimeProtocolValidationError(
            f"unknown runtime task result status {task_status!r}; expected one of: {allowed}",
            field="task_result.status",
            value=task_status,
        ) from exc


def validate_runtime_result_for_event(
    event_kind: str,
    result: Mapping[str, Any],
    *,
    terminal: bool | None = None,
    event_payload: Mapping[str, Any] | None = None,
) -> None:
    """Ensure a result agrees with the event carrying it."""

    status = result.get("status")
    allowed = _EVENT_RESULT_STATUSES.get(event_kind)
    if allowed is None:
        if terminal:
            raise RuntimeProtocolValidationError(
                f"terminal event {event_kind!r} cannot carry a runtime result",
                field="event.kind",
                value=event_kind,
            )
        raise RuntimeProtocolValidationError(
            f"non-terminal event {event_kind!r} cannot carry a runtime result",
            field="event.kind",
            value=event_kind,
        )
    if status not in allowed:
        expected = ", ".join(sorted(allowed))
        raise RuntimeProtocolValidationError(
            f"runtime result status {status!r} is inconsistent with event {event_kind!r}; expected: {expected}",
            field="status",
            value=status,
        )
    expected_terminal = event_kind in {
        RuntimeEventKind.RUN_COMPLETED.value,
        RuntimeEventKind.RUN_CLARIFICATION.value,
        RuntimeEventKind.RUN_FAILED.value,
        RuntimeEventKind.RUN_CANCELLED.value,
    }
    if terminal is not None and terminal != expected_terminal:
        raise RuntimeProtocolValidationError(
            f"terminal flag is inconsistent with event {event_kind!r}",
            field="event.terminal",
            value=terminal,
        )
    payload_status = (event_payload or {}).get("status")
    if payload_status is not None and payload_status != status:
        raise RuntimeProtocolValidationError(
            f"event payload status {payload_status!r} does not match result status {status!r}",
            field="event.payload.status",
            value=payload_status,
        )
