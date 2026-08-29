"""Durable repeatable budget tranches for product-owned agent tasks."""

from __future__ import annotations

from typing import Any, Mapping


RESEARCH_BUDGET_KEYS = ("model_calls", "model_tokens", "tool_calls", "elapsed_active_ms")
LIMIT_KEYS = {
    "model_calls": "max_model_calls",
    "model_tokens": "max_model_tokens",
    "tool_calls": "max_tool_calls",
    "elapsed_active_ms": "max_active_runtime_ms",
}
DEFAULT_LIMITS = {
    "model_calls": 10_000,
    "model_tokens": 500_000,
    "tool_calls": 100,
    "elapsed_active_ms": 3_600_000,
}


def tranche_limits(limits: Mapping[str, Any] | None) -> dict[str, int]:
    source = limits if isinstance(limits, Mapping) else {}
    return {
        key: max(1, int(source.get(key) or source.get(LIMIT_KEYS[key]) or default))
        for key, default in DEFAULT_LIMITS.items()
    }


def initial_budget_state(limits: Mapping[str, Any] | None) -> dict[str, Any]:
    zero = {key: 0 for key in RESEARCH_BUDGET_KEYS}
    return {
        "tranche_index": 1,
        "tranche_limits": tranche_limits(limits),
        "tranche_usage": dict(zero),
        "lifetime_usage": {**zero, "subagent_attempts": 0, "artifact_bytes": 0},
        "boundary": None,
    }


def normalize_budget_state(value: Mapping[str, Any] | None, limits: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(value or {})
    if isinstance(source.get("tranche_usage"), Mapping) and isinstance(source.get("lifetime_usage"), Mapping):
        zero = {key: 0 for key in RESEARCH_BUDGET_KEYS}
        state = {
            "tranche_index": 1,
            "tranche_limits": tranche_limits(limits),
            "tranche_usage": dict(zero),
            "lifetime_usage": {**zero, "subagent_attempts": 0, "artifact_bytes": 0},
            "boundary": None,
        }
        state.update(source)
        state["tranche_limits"] = tranche_limits(state.get("tranche_limits") or limits)
        state["tranche_usage"] = {
            key: max(0, int((source.get("tranche_usage") or {}).get(key) or 0))
            for key in RESEARCH_BUDGET_KEYS
        }
        lifetime = dict(source.get("lifetime_usage") or {})
        state["lifetime_usage"] = {
            **lifetime,
            **{key: max(0, int(lifetime.get(key) or 0)) for key in RESEARCH_BUDGET_KEYS},
        }
        return state
    # Existing JSON rows used flat counters. Preserve their accounting while
    # materializing the authoritative tranche shape on the next write.
    state = initial_budget_state(limits)
    for key in RESEARCH_BUDGET_KEYS:
        amount = max(0, int(source.get(key) or 0))
        state["tranche_usage"][key] = amount
        state["lifetime_usage"][key] = amount
    for key in ("subagent_attempts", "artifact_bytes"):
        state["lifetime_usage"][key] = max(0, int(source.get(key) or 0))
    return state


def exhausted_dimensions(state: Mapping[str, Any]) -> list[str]:
    usage = state.get("tranche_usage") if isinstance(state.get("tranche_usage"), Mapping) else {}
    limits = state.get("tranche_limits") if isinstance(state.get("tranche_limits"), Mapping) else {}
    return [key for key in RESEARCH_BUDGET_KEYS if int(usage.get(key) or 0) >= int(limits.get(key) or 1)]


def reset_tranche(state: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(state)
    result["tranche_index"] = max(1, int(result.get("tranche_index") or 1)) + 1
    result["tranche_usage"] = {key: 0 for key in RESEARCH_BUDGET_KEYS}
    result["boundary"] = None
    return result
