from langgraph_runtime.agent.canvas_layout_skills import (
    canvas_emit_enabled,
    canvas_publish_synthesis_enabled,
)


def test_canvas_publish_synthesis_requires_explicit_canvas_intent():
    base = {"allowed_tool_ids": ["research_canvas_publish"], "question": "What is the BLEU score?"}
    assert canvas_emit_enabled(base["allowed_tool_ids"]) is True
    assert canvas_publish_synthesis_enabled(base) is False
    assert canvas_publish_synthesis_enabled({**base, "question": "Build a timeline canvas of key events"}) is True
    assert canvas_publish_synthesis_enabled({**base, "task_orchestration": {"publish_canvas": True}}) is True
