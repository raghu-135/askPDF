"""Helpers for child runs that inherit parent context."""

from __future__ import annotations

from typing import Any, Mapping


def inherited_context_from_run_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(metadata, Mapping):
        return None
    window = metadata.get("inherited_context_window")
    budget = metadata.get("inherited_context_budget")
    if window is None and budget is None:
        return None
    payload: dict[str, Any] = {}
    if window is not None:
        payload["context_window"] = window
    if budget is not None:
        payload["context_budget"] = budget
    return payload or None