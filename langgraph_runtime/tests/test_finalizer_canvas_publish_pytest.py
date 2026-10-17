from unittest.mock import AsyncMock, patch

import pytest

from langgraph_runtime.workflows.answer_nodes import finalizer_node


@pytest.mark.asyncio
async def test_finalizer_surfaces_required_canvas_publish_failure():
    draft = {"layout": "matrix", "title": "Summary", "claims": [{"text": "x", "source_ids": ["doc:" + "a" * 32 + ":c-1"]}]}
    state = {
        "agent_task_id": "task-1",
        "final_answer": "Research summary in chat.",
        "task_canvas_draft": draft,
        "task_canvas_publish_required": True,
        "task_incomplete_reasons": [],
        "task_result_warnings": [],
    }
    with patch(
        "langgraph_runtime.workflows.answer_nodes.publish_deep_research_canvas_draft",
        new=AsyncMock(
            return_value={
                "ok": False,
                "error": {
                    "code": "publish_canvas_unknown_chunk",
                    "message": "could not resolve doc:…:c-0 (chunk row not found)",
                },
            },
        ),
    ):
        result = await finalizer_node(state, {"configurable": {}})

    assert result.get("task_canvas_published") is not True
    assert result.get("task_canvas_publish_failed", {}).get("code") == "publish_canvas_unknown_chunk"
    warning_codes = [item.get("code") for item in result.get("task_result_warnings") or []]
    assert "research_canvas_publish_failed" in warning_codes
    assert "research_canvas_unpublished" in (result.get("task_incomplete_reasons") or [])
