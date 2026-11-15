"""Shared flags for web worker / live external research (search or browser workbench)."""

from __future__ import annotations

import re
from typing import Mapping

BROWSER_TOOL_CONTRACT_IDS = frozenset({
    "browser_navigate",
    "browser_snapshot",
    "browser_act",
    "browser_read",
    "browser_save_to_thread",
})

EXTERNAL_RESEARCH_REQUEST_RE = re.compile(
    r"\b("
    r"browser(?:\s+agent)?|workbench\s+browser|shared\s+browser|"
    r"search\s+the\s+web|live\s+web|google\s+scholar|arxiv|"
    r"similar\s+papers?|related\s+papers?|find\s+papers?|"
    r"look\s+up\s+online|on\s+the\s+internet|online\s+search"
    r")\b",
    re.IGNORECASE,
)

AGENT_RUNTIME_META_RE = re.compile(
    r"(?:"
    r"\bwhat\s+tools?\b"
    r"|(?:what|which)\s+capabilit"
    r"|(?:what|which)\s+agents?\b"
    r"|(?:are|were)\s+you\s+able\s+to\b"
    r"|(?:do|can|could)\s+you\s+call\b"
    r"|(?:what|which)\s+workflows?\b"
    r"|(?:list|show)\s+(?:your\s+)?tools?\b"
    r")",
    re.IGNORECASE,
)


def _flag(mapping: Mapping[str, object] | None, key: str) -> bool:
    if not isinstance(mapping, Mapping):
        return False
    return bool(mapping.get(key))


def web_worker_enabled(state_or_settings: Mapping[str, object] | None) -> bool:
    return _flag(state_or_settings, "use_web_search") or _flag(state_or_settings, "use_browser_access")


def web_worker_enabled_flags(*, use_web_search: bool = False, use_browser_access: bool = False) -> bool:
    return bool(use_web_search) or bool(use_browser_access)


def question_requests_external_research(question: object) -> bool:
    return bool(EXTERNAL_RESEARCH_REQUEST_RE.search(str(question or "")))


def question_is_agent_runtime_meta(question: object) -> bool:
    return bool(AGENT_RUNTIME_META_RE.search(str(question or "")))


def attachment_document_scope_bypass(state: Mapping[str, object] | None) -> bool:
    """Skip attachment-only scope gates when live research or runtime meta is intended."""

    if not isinstance(state, Mapping):
        return False
    question = str(state.get("question") or "")
    if question_is_agent_runtime_meta(question):
        return True
    return web_worker_enabled(state) and question_requests_external_research(question)