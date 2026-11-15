"""Product-layer normalization for clarification terminal runs."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from app.db.enums import AgentRunStatus
CLARIFICATION_REQUIRED_STATUS = "clarification_required"


def clarification_options_from_result(result: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(result, Mapping):
        return []
    direct = result.get("clarification_options")
    if isinstance(direct, list) and direct:
        return [str(item).strip() for item in direct if str(item).strip()]
    nested = result.get("clarification")
    if isinstance(nested, Mapping):
        options = nested.get("options")
        if isinstance(options, list) and options:
            return [str(item).strip() for item in options if str(item).strip()]
    return []


def is_clarification_terminal_result(result: Mapping[str, Any] | None, *, status: Optional[str] = None) -> bool:
    if not isinstance(result, Mapping):
        return False
    if len(clarification_options_from_result(result)) >= 2:
        return True
    normalized = str(status or result.get("status") or "").strip().lower()
    return normalized in {
        CLARIFICATION_REQUIRED_STATUS,
        AgentRunStatus.CLARIFICATION.value,
        "clarification",
    }


def normalize_clarification_terminal_status(
    status: Optional[str],
    result: Mapping[str, Any] | None,
) -> str:
    if is_clarification_terminal_result(result, status=status):
        return CLARIFICATION_REQUIRED_STATUS
    if status:
        return status
    return AgentRunStatus.COMPLETED.value