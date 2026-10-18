from runtime_protocol.research_gaps import (
    dedupe_critic_issues,
    filter_false_canvas_absence_limitations,
    filter_stale_index_readiness_gaps,
    gap_display_line,
    strip_false_canvas_limitations_block,
    web_evidence_plan_gaps,
)
from runtime_protocol.task_results import normalize_runtime_task_result


def test_filter_stale_index_readiness_gaps_with_paren_citation_text():
    gaps = ["Document index is not ready for this thread."]
    filtered = filter_stale_index_readiness_gaps(
        gaps,
        manifest=[],
        evidence_texts=["Mara Quill (zephyr.pdf, page 1) sold 48,200 croissants."],
    )
    assert filtered == []


def test_filter_stale_index_readiness_gaps_drops_not_being_ready_wording():
    gaps = ["Inability to extract figures because the document index not being ready."]
    manifest = [{"source_id": "doc:abc:1", "file_hash": "abc", "page_number": 1}]
    filtered = filter_stale_index_readiness_gaps(gaps, manifest=manifest, evidence_texts=[])
    assert filtered == []


def test_dedupe_critic_issues_topic_overlap():
    gaps = ["The document index is not ready for this thread."]
    issues = ["document index not ready", "Unique finding"]
    deduped = dedupe_critic_issues(issues, gaps)
    assert "document index not ready" not in deduped
    assert deduped == ["Unique finding"]


def test_filter_stale_index_readiness_gaps_when_manifest_ready():
    gaps = [
        "Document index is not ready for this thread.",
        "Web benchmarks or live web evidence were requested in the objective, but web search is off for this task.",
    ]
    manifest = [{"source_id": "doc:abc:1", "file_hash": "abc", "page_number": 1}]
    filtered = filter_stale_index_readiness_gaps(gaps, manifest=manifest, evidence_texts=[])
    assert "Document index is not ready" not in " ".join(filtered)
    assert any("web search is off" in value.lower() for value in filtered)


def test_filter_stale_index_gaps_kept_without_document_evidence():
    gaps = ["Document index is not ready for this thread."]
    filtered = filter_stale_index_readiness_gaps(gaps, manifest=[], evidence_texts=[])
    assert filtered == gaps


def test_gap_display_line_prefers_message_on_structured_gap():
    assert gap_display_line({"code": "MISSING_SALES_FIGURES", "message": "Sales figures were not found."}) == (
        "Sales figures were not found."
    )


def test_normalize_runtime_task_result_structured_gaps():
    result = normalize_runtime_task_result({
        "status": "completed_with_warnings",
        "text": "Partial answer",
        "gaps": [{"code": "MISSING_SALES_FIGURES", "message": "Sales figures were not found."}],
    })
    assert result.gaps == ("Sales figures were not found.",)


def test_strip_false_canvas_limitations_block_removes_retest_variant_wording():
    text = (
        "Evidence summary.\n\n"
        "Limitations: No research canvas content present in the draft answer.; "
        "No document citations provided for any claims.; "
        "Does not follow the instruction to use publish canvas tool."
    )
    assert strip_false_canvas_limitations_block(text) == "Evidence summary."


def test_finalize_canvas_answer_prose_never_returns_empty_when_published():
    from runtime_protocol.research_gaps import finalize_canvas_answer_prose

    limitations_only = (
        "Limitations: No draft answer content present for evaluation; "
        "No citations or document references are provided in the draft answer."
    )
    cleaned = finalize_canvas_answer_prose(limitations_only, canvas_published=True)
    assert cleaned
    assert "Limitations" not in cleaned


def test_canvas_publish_outcome_reads_state_flags():
    from runtime_protocol.research_gaps import canvas_publish_outcome

    assert canvas_publish_outcome({"task_canvas_published": True}) == "published"
    assert canvas_publish_outcome({"task_canvas_publish_failed": {"code": "x"}}) == "failed"
    assert canvas_publish_outcome({}) == "not_attempted"


def test_filter_false_canvas_absence_limitations_rejects_evaluator_draft_denials():
    issues = [
        "No draft answer content present for evaluation",
        "No claims or data about WMT 2014 BLEU results are included",
        "No citations or document references are provided in the draft answer",
        "The user requested a research canvas using the publish canvas tool, but no such output is given",
    ]
    assert filter_false_canvas_absence_limitations(issues) == []


def test_strip_false_canvas_limitations_block_removes_published_canvas_denials():
    text = (
        "Summary of the paper.\n\n"
        "Limitations: No research canvas content provided; No document citations for claims; "
        "Does not follow instruction to use publish canvas tool"
    )
    cleaned = strip_false_canvas_limitations_block(text)
    assert cleaned == "Summary of the paper."
    assert filter_false_canvas_absence_limitations([
        "No research canvas content provided",
        "No document citations for claims",
        "Does not follow instruction to use publish canvas tool",
    ]) == []


def test_web_evidence_plan_gaps_respects_guidance_waiver():
    assert web_evidence_plan_gaps(
        "Compare using web benchmarks",
        "off",
        "Skip web benchmarks and use the PDF only.",
    ) == []
