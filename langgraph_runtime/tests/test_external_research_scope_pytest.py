"""Scope bypass when live web or browser tools are enabled (flag-based, not question wording)."""

from langgraph_runtime.workflows.document_scope import attachment_retrieval_scope_decline
from langgraph_runtime.workflows.external_research import (
    attachment_document_scope_bypass,
    web_worker_enabled,
)


def test_web_worker_enabled_follows_flags():
    assert web_worker_enabled({"use_web_search": True}) is True
    assert web_worker_enabled({"use_browser_access": True}) is True
    assert web_worker_enabled({}) is False


def test_attachment_document_scope_bypass_when_browser_enabled():
    state = {"use_browser_access": True, "question": "Any wording should not matter."}
    assert attachment_document_scope_bypass(state) is True
    assert attachment_retrieval_scope_decline(state) is None


def test_attachment_document_scope_bypass_off_allows_decline():
    state = {
        "use_browser_access": False,
        "use_web_search": False,
        "question": "Summarize chapter 1.",
        "pre_fetch_bundle": {"document_evidence_text": ""},
        "document_sources": ["doc-1"],
    }
    assert attachment_document_scope_bypass(state) is False