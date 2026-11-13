from __future__ import annotations

from typing import Any, Mapping

from langgraph_runtime.workflows.document_scope import (
    DOCUMENT_SCOPE_DECLINE_ANSWER,
    attachment_retrieval_scope_decline,
    thread_has_attached_documents,
)
from langgraph_runtime.workflows.enums import RouterRoute


def _prefetch_has_non_document_evidence(state: Mapping[str, Any]) -> bool:
    bundle = state.get("pre_fetch_bundle") or {}
    for key in ("web_search_evidence_text", "web_evidence_text"):
        if str(bundle.get(key) or "").strip():
            return True
    return False


def router_attachment_scope_decline(state: Mapping[str, Any], route: str | None) -> str | None:
    """Decline when uploaded documents exist but pre-fetch has no supporting evidence."""

    if route not in {RouterRoute.CLARIFY.value, RouterRoute.DIRECT.value}:
        return None
    if not thread_has_attached_documents(state):
        return None
    score_decline = attachment_retrieval_scope_decline(state)
    if score_decline:
        return score_decline
    if _prefetch_has_non_document_evidence(state):
        return None
    bundle = state.get("pre_fetch_bundle") or {}
    if not str(bundle.get("document_evidence_text") or "").strip():
        return DOCUMENT_SCOPE_DECLINE_ANSWER
    return None
