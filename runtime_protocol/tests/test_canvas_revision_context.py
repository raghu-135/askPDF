import copy

from runtime_protocol.canvas_revision_context import (
    build_revision_snapshot,
    format_quoted_canvas_revisions_for_prompt,
    merge_canvas_supersede_spec,
    merge_publish_canvas_draft,
)


def _matrix_spec_with_five_claims():
    rows = []
    citations = []
    for index in range(5):
        url = f"https://example.com/source-{index}"
        rows.append([
            {"text": f"Claim {index}"},
            {"text": f"Source label {index}"},
            {"text": "supported", "tone": "supported"},
            {"text": ""},
        ])
        citations.append({"kind": "web", "label": f"Source label {index}", "url": url})
    return {
        "schema_version": 1,
        "title": "Baseline canvas",
        "summary": "Original summary",
        "sections": [
            {
                "title": "Overview",
                "blocks": [
                    {"type": "stat", "value": "5", "label": "Claims reviewed", "tone": "info"},
                ],
            },
            {
                "title": "Evidence matrix",
                "blocks": [
                    {
                        "type": "table",
                        "caption": "Evidence matrix",
                        "headers": ["Claim", "Source", "Support", "Notes"],
                        "rows": rows,
                    }
                ],
            },
            {
                "title": "Sources",
                "blocks": [{"type": "sources", "citations": citations}],
            },
        ],
    }


def test_build_revision_snapshot_preserves_claim_and_source_counts():
    spec = _matrix_spec_with_five_claims()
    snapshot = build_revision_snapshot(
        canvas_id="canvas-quoted",
        lineage_head_id="canvas-head",
        title="Baseline canvas",
        spec=spec,
    )
    assert snapshot["claim_count"] == 5
    assert snapshot["source_count"] == 5
    assert len(snapshot["publish_canvas_draft"]["claims"]) == 5


def test_merge_publish_canvas_draft_restores_dropped_claims_on_supersede():
    spec = _matrix_spec_with_five_claims()
    baseline = build_revision_snapshot(
        canvas_id="canvas-quoted",
        lineage_head_id="canvas-head",
        title="Baseline canvas",
        spec=spec,
    )["publish_canvas_draft"]
    incoming = {
        "layout": "matrix",
        "title": "Updated title",
        "summary": "New summary at top",
        "claims": [
            {
                "text": "Claim 0",
                "source_ids": [f"web:https://example.com/source-0"],
            }
        ],
    }
    merged = merge_publish_canvas_draft(baseline, incoming)
    assert merged["summary"] == "New summary at top"
    assert len(merged["claims"]) == 5


def test_merge_canvas_supersede_spec_promotes_changed_summary_field():
    baseline = _matrix_spec_with_five_claims()
    baseline["summary"] = "Original opening"
    incoming = {
        **copy.deepcopy(baseline),
        "summary": "Short summary at the top for readers.",
    }
    merged = merge_canvas_supersede_spec(baseline, incoming)
    titles = [section["title"] for section in merged["sections"]]
    assert len(titles) >= 4
    matrix = next(section for section in merged["sections"] if section["title"] == "Evidence matrix")
    table = next(block for block in matrix["blocks"] if block["type"] == "table")
    assert len(table["rows"]) == 5


def test_merge_canvas_supersede_spec_adds_new_section_and_preserves_claims():
    baseline = _matrix_spec_with_five_claims()
    incoming = {
        "schema_version": 1,
        "title": "Updated",
        "summary": "Top summary",
        "sections": [
            {
                "title": "Overview",
                "blocks": [
                    {"type": "stat", "value": "5", "label": "Claims reviewed", "tone": "info"},
                    {"type": "markdown", "text": "Original summary"},
                    {"type": "markdown", "text": "Top summary"},
                ],
            },
            *baseline["sections"][1:],
        ],
    }
    merged = merge_canvas_supersede_spec(baseline, incoming)
    titles = [section["title"] for section in merged["sections"]]
    assert len(titles) >= 4
    assert titles[0] != "Overview"
    matrix = next(section for section in merged["sections"] if section["title"] == "Evidence matrix")
    table = next(block for block in matrix["blocks"] if block["type"] == "table")
    assert len(table["rows"]) == 5


def test_canvas_prompt_includes_revision_baseline():
    spec = _matrix_spec_with_five_claims()
    snapshot = build_revision_snapshot(
        canvas_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        lineage_head_id="aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
        title="Baseline canvas",
        spec=spec,
    )
    section = format_quoted_canvas_revisions_for_prompt([snapshot])
    assert "Quoted canvas revision baseline" in section
    assert "Claim 4" in section
    assert "Claims: 5" in section