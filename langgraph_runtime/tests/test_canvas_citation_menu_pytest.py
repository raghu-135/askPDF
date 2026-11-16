from langgraph_runtime.workflows.canvas_citation_menu import (
    canvas_prompt_sections,
    format_current_canvases_section,
)


def test_current_canvases_section_reads_prefetch_bundle():
    state = {
        "pre_fetch_bundle": {
            "current_canvases": [{"id": "canvas-1", "title": "Compare papers"}],
        },
    }
    section = format_current_canvases_section(state)
    assert "canvas-1" in section
    assert "Compare papers" in section


def test_canvas_prompt_sections_includes_quoted_canvas_revision_baseline():
    state = {
        "pre_fetch_bundle": {
            "quoted_canvas_revisions": [
                {
                    "canvas_id": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                    "lineage_head_id": "aaaaaaaa-aaaa-aaaa-8aaa-aaaaaaaaaaaa",
                    "claim_count": 3,
                    "source_count": 3,
                    "publish_canvas_draft": {
                        "layout": "matrix",
                        "title": "Existing",
                        "summary": "Keep me",
                        "claims": [
                            {"text": "A", "source_ids": ["web:https://example.com/a"]},
                            {"text": "B", "source_ids": ["web:https://example.com/b"]},
                            {"text": "C", "source_ids": ["web:https://example.com/c"]},
                        ],
                    },
                }
            ],
        },
    }
    combined = canvas_prompt_sections(state)
    assert "Quoted canvas revision baseline" in combined
    assert "Keep me" in combined


def test_canvas_prompt_sections_includes_citation_menu_from_evidence():
    state = {
        "evidence_packets": [
            {
                "source_ids": ["web:https://example.com/paper"],
                "display": {"title": "Example paper"},
                "locator": {"sentence_ids": [2, 3]},
            },
        ],
    }
    combined = canvas_prompt_sections(state)
    assert "web:https://example.com/paper" in combined
    assert "Citation menu" in combined
