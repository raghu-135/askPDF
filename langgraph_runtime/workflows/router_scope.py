from __future__ import annotations

from typing import Any, Mapping

from langgraph_runtime.workflows.corrective_nodes import DOCUMENT_SCOPE_DECLINE_ANSWER
from langgraph_runtime.workflows.enums import RouterRoute


def _attached_documents(state: Mapping[str, Any]) -> bool:
    bundle = state.get("pre_fetch_bundle") or {}
    if state.get("document_sources"):
        return True
    documents = bundle.get("documents")
    return bool(documents) if isinstance(documents, list) else bool(documents)


def _prefetch_has_document_evidence(state: Mapping[str, Any]) -> bool:
    bundle = state.get("pre_fetch_bundle") or {}
    for key in ("document_evidence_text", "web_search_evidence_text", "web_evidence_text"):
        if str(bundle.get(key) or "").strip():
            return True
    return False


def router_attachment_scope_decline(state: Mapping[str, Any], route: str | None) -> str | None:
    """Decline when uploaded documents exist but pre-fetch has no supporting evidence."""

    if route not in {RouterRoute.CLARIFY.value, RouterRoute.DIRECT.value}:
        return None
    if not _attached_documents(state):
        return None
    if _prefetch_has_document_evidence(state):
        return None
    return DOCUMENT_SCOPE_DECLINE_ANSWER
