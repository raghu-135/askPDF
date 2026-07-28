"""Canonical durable-memory consent settings and local user scope."""

from __future__ import annotations

from typing import Any, Dict, Mapping


LOCAL_USER_MEMORY_SCOPE_ID = "default"

DEFAULT_PROJECT_MEMORY_SETTINGS = {
    "project_reads_user_memory": False,
}

DEFAULT_THREAD_MEMORY_SETTINGS = {
    "thread_reads_project_memory": True,
    "thread_reads_user_memory": False,
}


def normalize_project_memory_settings(settings_json: Any) -> Dict[str, bool]:
    settings = settings_json if isinstance(settings_json, Mapping) else {}
    raw_memory = settings.get("memory")
    memory = raw_memory if isinstance(raw_memory, Mapping) else {}
    return {
        "project_reads_user_memory": bool(
            memory.get(
                "project_reads_user_memory",
                DEFAULT_PROJECT_MEMORY_SETTINGS["project_reads_user_memory"],
            )
        ),
    }


def normalize_thread_memory_settings(settings: Any) -> Dict[str, bool]:
    settings_mapping = settings if isinstance(settings, Mapping) else {}
    raw_memory = settings_mapping.get("memory")
    memory = raw_memory if isinstance(raw_memory, Mapping) else {}

    reads_user_memory = bool(
        memory.get(
            "thread_reads_user_memory",
            DEFAULT_THREAD_MEMORY_SETTINGS["thread_reads_user_memory"],
        )
    )
    if "global_memory_enabled" in memory:
        reads_user_memory = reads_user_memory and bool(memory.get("global_memory_enabled"))

    return {
        "thread_reads_project_memory": bool(
            memory.get(
                "thread_reads_project_memory",
                DEFAULT_THREAD_MEMORY_SETTINGS["thread_reads_project_memory"],
            )
        ),
        "thread_reads_user_memory": reads_user_memory,
    }


def merge_project_settings_json(
    current: Any,
    updates: Any = None,
) -> Dict[str, Any]:
    """Merge project settings while persisting only canonical memory keys."""

    merged = dict(current) if isinstance(current, Mapping) else {}
    if isinstance(updates, Mapping):
        merged.update(updates)

    memory_source = merged
    if isinstance(updates, Mapping) and isinstance(updates.get("memory"), Mapping):
        prior_memory = current.get("memory") if isinstance(current, Mapping) else {}
        merged_memory = dict(prior_memory) if isinstance(prior_memory, Mapping) else {}
        merged_memory.update(updates["memory"])
        memory_source = {**merged, "memory": merged_memory}

    merged["memory"] = normalize_project_memory_settings(memory_source)
    return merged
