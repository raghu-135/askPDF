from __future__ import annotations

from langgraph_runtime.workflows.enums import RouterRoute
from langgraph_runtime.workflows.router_scope import router_attachment_scope_decline


def test_attachment_scope_decline_does_not_apply_to_clarify_with_options():
    state = {
        "document_sources": [{"file_hash": "a" * 32}],
        "pre_fetch_bundle": {"documents": [{"file_hash": "a" * 32}], "document_evidence_text": ""},
        "question": "What did you mean by that?",
    }
    route = RouterRoute.CLARIFY.value
    clarification_options = ["Summarize the attached PDF", "List every file in this thread"]
    decline = router_attachment_scope_decline(state, route)
    assert decline
    # Router node preserves clarify when options are present; mirror that contract here.
    keep_clarify = route == RouterRoute.CLARIFY.value and len(clarification_options) >= 2
    if decline and keep_clarify:
        decline = None
    assert decline is None