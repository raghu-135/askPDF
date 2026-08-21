"""Deterministic askPDF Hermes definition to managed-profile resolution."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping
from app.runtime.hermes_compatibility import (
    HERMES_DEFINITION_VERSION, HERMES_EXTERNAL_PROFILE, HERMES_OFFLINE_PROFILE,
    HERMES_PROFILE_VERSION, HERMES_SUPPORTED_DEFINITION_VERSIONS,
)
HERMES_BASE_TOOL_IDS = (
    "get_thread_shape",
    "search_document_by_id",
    "search_documents",
    "search_durable_memory",
    "search_thread_conversation_history",
    "search_thread_events",
)
HERMES_EXTERNAL_TOOL_IDS = (
    "arxiv",
    "pubmed",
    "search_web",
    "semantic_scholar",
    "stack_exchange",
    "wikipedia",
    "wikidata",
    "yahoo_finance_news",
)
HERMES_RESEARCH_TOOL_IDS = frozenset(HERMES_BASE_TOOL_IDS + HERMES_EXTERNAL_TOOL_IDS)
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
    definition_version = int(spec.get("definition_version") or 0)
    if definition_version not in HERMES_SUPPORTED_DEFINITION_VERSIONS:
        raise ValueError(f"Hermes definition_version must be one of {sorted(HERMES_SUPPORTED_DEFINITION_VERSIONS)}")
    config = spec.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("Hermes definitions require config")
    _reject_secrets(config)
    configured_tools = set(str(item) for item in config.get("allowed_tool_ids") or [])
    unknown_tools = configured_tools - HERMES_RESEARCH_TOOL_IDS
    if unknown_tools:
        raise ValueError(f"Hermes definitions contain unsupported research tools: {', '.join(sorted(unknown_tools))}")
    external_enabled = bool(config.get("use_web_search", False))
    permitted = set(HERMES_BASE_TOOL_IDS)
    if external_enabled:
        permitted.update(HERMES_EXTERNAL_TOOL_IDS)
    tools = tuple(sorted(configured_tools & permitted))
    runtime_profile = HERMES_EXTERNAL_PROFILE if external_enabled else HERMES_OFFLINE_PROFILE
    skills = tuple(sorted(set(str(item) for item in config.get("skills") or [])))
    profile = {
        "profile_version": HERMES_PROFILE_VERSION if definition_version == HERMES_DEFINITION_VERSION else 1,
        "instructions": str(config.get("system_prompt") or ""),
        "mcp": {
            "server": str(config.get("mcp_server") or ""),
            "allowed_tool_ids": list(tools),
            "runtime_profile": runtime_profile,
        },
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
