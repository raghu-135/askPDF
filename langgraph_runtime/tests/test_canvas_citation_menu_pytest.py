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
