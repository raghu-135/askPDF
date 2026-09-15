from app.agent.canvas_layout_skills import (
    CANVAS_LAYOUT_SKILL_IDS,
    apply_canvas_layout_skills,
    canvas_emit_enabled,
    canvas_layout_skills_section,
)


def test_canvas_layout_skills_require_slice_3_admission():
    assert canvas_emit_enabled(None) is False
    assert canvas_emit_enabled(["search_knowledge"]) is False
    assert canvas_emit_enabled(["research_canvas_publish"]) is True
    assert canvas_emit_enabled(["publish_canvas"]) is True
    assert canvas_layout_skills_section(["search_knowledge"]) == ""
    assert "canvas_compare_papers" in canvas_layout_skills_section(["publish_canvas"])
    assert "canvas_evidence_matrix" in canvas_layout_skills_section(["research_canvas_publish"])
    assert "canvas_timeline" in canvas_layout_skills_section(["publish_canvas"])


def test_apply_canvas_layout_skills_only_when_publish_is_admitted():
    stripped = apply_canvas_layout_skills(
        {
            "system_prompt": "Keep this.",
            "allowed_tool_ids": ["search_knowledge"],
            "skills": list(CANVAS_LAYOUT_SKILL_IDS),
        }
    )
    assert stripped["skills"] == []
    assert stripped["system_prompt"] == "Keep this."
    assert "canvas_compare_papers" not in stripped["system_prompt"]

    admitted = apply_canvas_layout_skills(
        {"system_prompt": "Keep this.", "allowed_tool_ids": ["publish_canvas"]}
    )
    assert list(CANVAS_LAYOUT_SKILL_IDS) == [item for item in admitted["skills"] if item in CANVAS_LAYOUT_SKILL_IDS]
    assert "Keep this." in admitted["system_prompt"]
    assert "canvas_compare_papers" in admitted["system_prompt"]
    assert "stat" in admitted["system_prompt"] and "table" in admitted["system_prompt"]
