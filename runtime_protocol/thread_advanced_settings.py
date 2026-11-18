"""Shared advanced thread settings: defaults, validation, merge, and field metadata."""

from __future__ import annotations

from typing import Any, Mapping

from runtime_protocol.context_budget import (
    ContextBudgetSettings,
    default_context_window_tokens,
    load_context_budget_settings,
    resolve_context_budget,
)
from runtime_protocol.coverage_iterations import (
    coverage_iteration_field_specs,
    load_coverage_iteration_settings,
    merge_coverage_iteration_overrides,
    resolve_coverage_iteration_settings,
)

_CONTEXT_BUDGET_KEYS = tuple(load_context_budget_settings().to_dict().keys())
_CONTEXT_WINDOW_BOUNDS = (256, 2_000_000)


def advanced_context_budget_defaults() -> dict[str, float]:
    return load_context_budget_settings().to_dict()


def advanced_context_window_default(environ: Mapping[str, str] | None = None) -> int:
    return default_context_window_tokens(environ)


def context_budget_field_specs() -> list[dict[str, Any]]:
    defaults = load_context_budget_settings()
    specs: list[dict[str, Any]] = []
    for key in _CONTEXT_BUDGET_KEYS:
        specs.append(
            {
                "key": key,
                "type": "ratio",
                "minimum": 0.0,
                "maximum": 1.0,
                "exclusive_minimum": True,
                "default": float(getattr(defaults, key)),
                "env_var": f"CONTEXT_{key.upper()}",
            }
        )
    return specs


def advanced_settings_field_specs() -> list[dict[str, Any]]:
    """Single catalog for API defaults and UI Advanced section."""
    return [
        {
            "key": "context_window",
            "type": "integer",
            "minimum": _CONTEXT_WINDOW_BOUNDS[0],
            "maximum": _CONTEXT_WINDOW_BOUNDS[1],
            "default": advanced_context_window_default(),
            "env_var": "DEFAULT_TOKEN_BUDGET",
        },
        *context_budget_field_specs(),
        *coverage_iteration_field_specs(),
    ]


def _clamp_context_window(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("context_window must be an integer") from exc
    low, high = _CONTEXT_WINDOW_BOUNDS
    return max(low, min(high, parsed))


def merge_context_budget_overrides(
    stored: Mapping[str, Any] | None,
    updates: Mapping[str, Any] | None,
) -> dict[str, float]:
    prior = dict(stored) if isinstance(stored, Mapping) else {}
    if not isinstance(updates, Mapping):
        return prior
    patch: dict[str, Any] = {}
    for key in _CONTEXT_BUDGET_KEYS:
        if key not in updates:
            continue
        value = updates[key]
        if value is None:
            prior.pop(key, None)
        else:
            patch[key] = value
    if patch:
        resolved = resolve_context_budget({**prior, **patch})
        for key in patch:
            prior[key] = float(getattr(resolved, key))
    return prior


def resolve_thread_advanced_settings(
    thread_settings: Mapping[str, Any] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fully resolved advanced values for runtime (env defaults + thread partials)."""
    source = thread_settings if isinstance(thread_settings, Mapping) else {}
    budget_overrides = source.get("context_budget") if isinstance(source.get("context_budget"), Mapping) else {}
    coverage_overrides = (
        source.get("coverage_iterations") if isinstance(source.get("coverage_iterations"), Mapping) else {}
    )
    budget: ContextBudgetSettings = resolve_context_budget(budget_overrides, environ=environ)
    coverage = resolve_coverage_iteration_settings(coverage_overrides, environ=environ)
    window_raw = source.get("context_window")
    context_window = (
        _clamp_context_window(window_raw)
        if window_raw is not None
        else advanced_context_window_default(environ)
    )
    return {
        "context_window": context_window,
        "context_budget": budget.to_dict(),
        "coverage_iterations": coverage.to_dict(),
    }


def merge_thread_advanced_storage(
    prior: Mapping[str, Any] | None,
    updates: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Persist only explicit overrides on the thread settings document."""
    merged: dict[str, Any] = {}
    base = prior if isinstance(prior, Mapping) else {}
    if not isinstance(updates, Mapping):
        updates = {}

    reset_window = updates.get("context_window") is None and "context_window" in updates
    if updates.get("context_window") is not None:
        merged["context_window"] = _clamp_context_window(updates["context_window"])

    if "context_budget" in updates:
        merged["context_budget"] = merge_context_budget_overrides(
            base.get("context_budget") if isinstance(base.get("context_budget"), Mapping) else {},
            updates.get("context_budget"),
        )

    if "coverage_iterations" in updates:
        merged["coverage_iterations"] = merge_coverage_iteration_overrides(
            base.get("coverage_iterations") if isinstance(base.get("coverage_iterations"), Mapping) else {},
            updates.get("coverage_iterations"),
        )

    for key in ("context_budget", "coverage_iterations"):
        if key in base and key not in merged:
            merged[key] = base[key]
        if key in merged and merged[key] in ({}, None):
            merged.pop(key, None)
    if not reset_window and "context_window" in base and "context_window" not in merged:
        merged["context_window"] = base["context_window"]

    return merged