"""Deployment-configured budgets shared by the Deep Agent runtimes."""

from __future__ import annotations

import os
import re
from typing import Any, Mapping


DEEP_AGENT_BUDGET_KEYS = frozenset({
    "max_model_calls",
    "max_model_tokens",
    "max_tool_calls",
    "max_active_runtime_ms",
    "max_duration_seconds",
    "max_output_chars",
    "max_event_count",
    "wake_limit_seconds",
    "subagent_timeout_ms",
    "dispatch_timeout_ms",
    "worker_timeout_ms",
    "web_worker_timeout_ms",
})

# These are the limits each framework can consume. An absent key is
# intentional: framework adapters must not infer support for another runtime's
# execution model.
DEEP_AGENT_FRAMEWORK_KEYS: dict[str, frozenset[str]] = {
    "langgraph": DEEP_AGENT_BUDGET_KEYS,
    "hermes": frozenset({
        "max_model_calls",
        "max_model_tokens",
        "max_tool_calls",
        "max_active_runtime_ms",
        "max_duration_seconds",
        "max_output_chars",
        "max_event_count",
        "wake_limit_seconds",
    }),
}

_ENV_SPECS: dict[str, tuple[str, int]] = {
    "max_model_calls": ("MAX_MODEL_CALLS", 1),
    "max_model_tokens": ("MAX_MODEL_TOKENS", 1),
    "max_tool_calls": ("MAX_TOOL_CALLS", 1),
    "max_active_runtime_ms": ("MAX_ACTIVE_RUNTIME_MS", 1),
    "max_duration_seconds": ("MAX_DURATION_MS", 1000),
    "max_output_chars": ("MAX_OUTPUT_CHARS", 1),
    "max_event_count": ("MAX_EVENT_COUNT", 1),
    "wake_limit_seconds": ("WAKE_LIMIT_SECONDS", 1),
    "subagent_timeout_ms": ("SUBAGENT_TIMEOUT_MS", 1),
    "dispatch_timeout_ms": ("DISPATCH_TIMEOUT_MS", 1),
    "worker_timeout_ms": ("WORKER_TIMEOUT_MS", 1),
    "web_worker_timeout_ms": ("WEB_WORKER_TIMEOUT_MS", 1),
}


def _env_key(framework: str | None, name: str) -> str:
    suffix = _ENV_SPECS[name][0]
    if framework:
        return f"DEEP_AGENT_{framework.upper()}_{suffix}"
    return f"DEEP_AGENT_{suffix}"


_ENV_REFERENCE = re.compile(r"^\$?\{?([A-Z][A-Z0-9_]*)\}?$")


def _raw_env(name: str, seen: frozenset[str] = frozenset()) -> str | None:
    if name in seen:
        raise ValueError(f"cyclic Deep Agent environment reference involving {name}")
    raw = os.getenv(name)
    if raw is None:
        return None
    match = _ENV_REFERENCE.match(raw.strip())
    if match and match.group(1) != name and match.group(1).startswith("DEEP_AGENT_"):
        return _raw_env(match.group(1), seen | {name})
    return raw


def _positive_env(name: str) -> int | None:
    raw = _raw_env(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def deep_agent_budgets(framework: str | None = None) -> dict[str, int]:
    """Return required deployment budgets for a framework."""

    normalized = framework.lower() if framework else None
    if normalized is not None and normalized not in DEEP_AGENT_FRAMEWORK_KEYS:
        raise ValueError(f"unsupported Deep Agent framework: {framework}")
    keys = DEEP_AGENT_FRAMEWORK_KEYS.get(normalized, DEEP_AGENT_BUDGET_KEYS)
    result: dict[str, int] = {}
    for name in keys:
        value = _positive_env(_env_key(normalized, name))
        if value is None:
            value = _positive_env(_env_key(None, name))
        if value is None:
            raise ValueError(f"{_env_key(normalized, name)} or {_env_key(None, name)} is required")
        divisor = _ENV_SPECS[name][1]
        result[name] = max(1, (value + divisor - 1) // divisor)
    return result


def apply_deep_agent_env_overrides(
    limits: Mapping[str, Any],
    framework: str,
) -> dict[str, Any]:
    """Apply deployment overrides without inventing unsupported fields."""

    result = dict(limits)
    for name, value in deep_agent_budgets(framework).items():
        framework_name = _env_key(framework, name)
        common_name = _env_key(None, name)
        if os.getenv(framework_name) is not None:
            result[name] = value
        elif os.getenv(common_name) is not None:
            result[name] = value
    return result


def configured_budget_value(config: Mapping[str, Any], name: str, framework: str) -> int:
    """Resolve an adapter limit from required deployment configuration."""

    budgets = deep_agent_budgets(framework)
    return budgets[name]
