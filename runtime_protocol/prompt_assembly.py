"""Window-aware prompt assembly helpers (workflow-internal caps from env)."""

from __future__ import annotations

from typing import Any, Mapping

from runtime_protocol.context_budget import load_context_budget_settings, usable_context_chars


def workflow_char_limit(
    context_window: int | None,
    *,
    ratio_key: str,
) -> int:
    """Char limit for a prompt slice using fleet env ratios, not per-run prefetch overrides."""

    return usable_context_chars(
        context_window,
        ratio_key=ratio_key,
        budget=load_context_budget_settings(),
    )


def truncate_for_workflow(text: str, context_window: int | None, *, ratio_key: str) -> str:
    limit = workflow_char_limit(context_window, ratio_key=ratio_key)
    value = str(text or "")
    return value[:limit] if limit > 0 else value