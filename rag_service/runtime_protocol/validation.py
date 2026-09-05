"""Strict validation for runtime result envelopes."""

from __future__ import annotations

from typing import Any, Mapping

from runtime_protocol.contracts import RuntimeTaskResultStatus


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
