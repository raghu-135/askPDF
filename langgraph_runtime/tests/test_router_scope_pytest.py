from __future__ import annotations

from langgraph_runtime.workflows.corrective_nodes import DOCUMENT_SCOPE_DECLINE_ANSWER
from langgraph_runtime.workflows.enums import RouterRoute
from langgraph_runtime.workflows.router_scope import router_attachment_scope_decline


def test_router_scope_decline_on_clarify_without_prefetch_evidence():
    state = {
        "document_sources": [{"file_hash": "a" * 32}],
        "pre_fetch_bundle": {"documents": [{"file_hash": "a" * 32}], "document_evidence_text": ""},
        "question": "What is the population of Tokyo?",
    }
    assert router_attachment_scope_decline(state, RouterRoute.CLARIFY.value) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_router_scope_decline_skips_when_prefetch_has_high_scoring_document_evidence():
    state = {
        "document_sources": [{"file_hash": "a" * 32, "score": 0.88}],
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_evidence_text": "Tokyo has 14 million residents in the metro area.",
            "document_sources": [{"file_hash": "a" * 32, "score": 0.88}],
        },
    }
    assert router_attachment_scope_decline(state, RouterRoute.CLARIFY.value) is None
