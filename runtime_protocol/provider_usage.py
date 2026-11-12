"""Provider-reported LLM usage helpers (OpenRouter-compatible)."""

from __future__ import annotations

from typing import Any, Mapping


def provider_cost_usd(usage: Mapping[str, Any] | None) -> float | None:
    """Return the provider-reported USD cost when present; never estimate from tokens."""

    if not isinstance(usage, Mapping):
        return None
    for key in ("cost", "provider_cost_usd", "total_cost"):
        raw = usage.get(key)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value < 0:
            return None
        return value
    return None


def usage_with_provider_cost(
    usage: Mapping[str, Any] | None,
    *,
    response_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge token usage with an optional provider cost, when the response includes one."""

    merged: dict[str, Any] = dict(usage or {})
    candidates: list[Mapping[str, Any]] = [merged]
    if isinstance(response_metadata, Mapping):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, Mapping):
            candidates.append(token_usage)
        top_level = response_metadata.get("usage")
        if isinstance(top_level, Mapping):
            candidates.append(top_level)
    for candidate in candidates:
        cost = provider_cost_usd(candidate)
        if cost is not None:
            merged["cost"] = cost
            break
    return merged
