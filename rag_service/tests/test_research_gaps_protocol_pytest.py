"""Ensure shared gap helpers live in runtime_protocol (rag-service image dependency)."""

from runtime_protocol.research_gaps import dedupe_critic_issues, filter_stale_index_readiness_gaps


def test_filter_stale_index_gaps_via_protocol_package():
    gaps = ["Document index is not ready for this thread."]
    filtered = filter_stale_index_readiness_gaps(
        gaps,
        manifest=[],
        evidence_texts=["Evidence (zephyr.pdf, page 1)"],
    )
    assert filtered == []


def test_dedupe_critic_issues_via_protocol_package():
    gaps = ["The document index is not ready for this thread."]
    issues = ["document index not ready", "Other"]
    assert dedupe_critic_issues(issues, gaps) == ["Other"]
