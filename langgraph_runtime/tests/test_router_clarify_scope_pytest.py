from __future__ import annotations

from langgraph_runtime.workflows.enums import RouterRoute
from langgraph_runtime.workflows.router_scope import router_attachment_scope_decline


def test_attachment_scope_decline_skips_clarify_route_with_attachments():
    state = {
        "document_sources": [{"file_hash": "a" * 32}],
        "pre_fetch_bundle": {"documents": [{"file_hash": "a" * 32}], "document_evidence_text": ""},
        "question": "What did you mean by that?",
    }
    assert router_attachment_scope_decline(state, RouterRoute.CLARIFY.value) is None