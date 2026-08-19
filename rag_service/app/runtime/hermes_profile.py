"""Deterministic askPDF Hermes definition to managed-profile resolution."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


HERMES_DEFINITION_VERSION = 1
HERMES_PROFILE_VERSION = 1
_SECRET_KEYS = frozenset({"api_key", "token", "secret", "password", "authorization", "credentials"})


def _reject_secrets(value: Any, path: str = "config") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _SECRET_KEYS or normalized.endswith(("_api_key", "_token", "_secret", "_password")):
                raise ValueError(f"Hermes definitions cannot persist credentials: {path}.{key}")
            _reject_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{path}[{index}]")


def resolve_hermes_profile(spec: Mapping[str, Any]) -> dict[str, Any]:
    if int(spec.get("definition_version") or 0) != HERMES_DEFINITION_VERSION:
        raise ValueError(f"Hermes definition_version must be {HERMES_DEFINITION_VERSION}")
    config = spec.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Hermes definitions require config")
    _reject_secrets(config)
    tools = tuple(sorted(set(str(item) for item in config.get("allowed_tool_ids") or [])))
    skills = tuple(sorted(set(str(item) for item in config.get("skills") or [])))
    profile = {
        "profile_version": HERMES_PROFILE_VERSION,
        "instructions": str(config.get("system_prompt") or ""),
        "mcp": {"server": str(config.get("mcp_server") or ""), "allowed_tool_ids": list(tools)},
        "model_policy": {"model": str(config.get("model") or ""), "provider": config.get("provider")},
        "skills": {"enabled": list(skills)},
        "memory": {"persistent": bool(config.get("allow_persistent_memory", False))},
        "delegation": {"enabled": bool(config.get("allow_subagents", False))},
        "limits": {
            "max_output_chars": int(config.get("max_output_chars") or 12000),
            "max_duration_seconds": int(config.get("max_duration_seconds") or 300),
            "max_event_count": int(config.get("max_event_count") or 200),
        },
        "context_window": int(config.get("context_window") or 0) or None,
        "task_policy": dict(config.get("task_policy") or {}),
    }
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"))
    return {**profile, "profile_id": hashlib.sha256(canonical.encode()).hexdigest()}
