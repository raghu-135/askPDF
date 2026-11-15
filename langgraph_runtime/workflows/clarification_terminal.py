"""Shared helpers for workflow clarification terminals (HITL choice card, not chat text)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from langgraph_runtime.workflows.planning import normalize_clarification_options

CLARIFICATION_REQUIRED_STATUS = "clarification_required"


def clarification_options_from_mapping(value: Mapping[str, Any] | None) -> List[str]:
    if not isinstance(value, Mapping):
        return []
    options = value.get("clarification_options")
    if isinstance(options, list) and options:
        return normalize_clarification_options(options)
    nested = value.get("clarification")
    if isinstance(nested, Mapping):
        nested_options = nested.get("options")
        if isinstance(nested_options, list) and nested_options:
            return normalize_clarification_options(nested_options)
    return []


def is_clarification_terminal_status(status: Any) -> bool:
    normalized = str(status or "").strip().lower()
    return normalized in {
        CLARIFICATION_REQUIRED_STATUS,
        "clarification",
    }


def clarification_terminal_active(
    payload: Mapping[str, Any] | None,
    *,
  minimum_options: int = 2,
) -> bool:
    options = clarification_options_from_mapping(payload or {})
    if len(options) >= minimum_options:
        return True
    return is_clarification_terminal_status((payload or {}).get("status"))


def clarification_terminal_state_update(
    options: Sequence[str],
    *,
    route_reason: Optional[str] = None,
) -> Dict[str, Any]:
    normalized = normalize_clarification_options(list(options))
    update: Dict[str, Any] = {
        "clarification_options": normalized,
        "final_answer": "",
        "answer": "",
        "status": CLARIFICATION_REQUIRED_STATUS,
    }
    if route_reason:
        update["route_reason"] = route_reason
    return update