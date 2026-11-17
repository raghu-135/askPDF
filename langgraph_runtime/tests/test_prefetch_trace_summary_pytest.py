from __future__ import annotations

from langgraph_runtime.workflows.evidence import prefetch_trace_summary


def test_prefetch_trace_summary_counts_without_full_text():
    summary = prefetch_trace_summary(
        {
            "document_evidence_text": "abc",
            "document_sources": [{"id": "1"}],
            "used_chat_ids": [],
        },
        context_budget={"document": 0.5, "recent_history": 0.1},
    )
    assert summary["slice_char_counts"]["document_evidence"] == 3
    assert summary["document_source_count"] == 1
    assert summary["context_budget"]["document"] == 0.5