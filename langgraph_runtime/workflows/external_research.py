"""Shared flags for web worker / live external research (search or browser workbench)."""

from __future__ import annotations

from typing import Mapping

BROWSER_TOOL_CONTRACT_IDS = frozenset({
    "browser_navigate",
    "browser_snapshot",
    "browser_act",
    "browser_read",
    "browser_save_to_thread",
})


def _flag(mapping: Mapping[str, object] | None, key: str) -> bool:
    if not isinstance(mapping, Mapping):
        return False
    return bool(mapping.get(key))


def web_worker_enabled(state_or_settings: Mapping[str, object] | None) -> bool:
    return _flag(state_or_settings, "use_web_search") or _flag(state_or_settings, "use_browser_access")


def web_worker_enabled_flags(*, use_web_search: bool = False, use_browser_access: bool = False) -> bool:
    return bool(use_web_search) or bool(use_browser_access)


def attachment_document_scope_bypass(state: Mapping[str, object] | None) -> bool:
    """Skip attachment-only scope gates when live web or browser tools are enabled for the run."""

    return web_worker_enabled(state)