from runtime_protocol.canvas_revision_context import (
    build_revision_snapshot,
    format_quoted_canvas_revisions_for_prompt,
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
    assert "supersedes_id" in section