from unittest.mock import AsyncMock, patch

import pytest

from langgraph_runtime.workflows.answer_nodes import finalizer_node


@pytest.mark.asyncio
async def test_finalizer_surfaces_required_canvas_publish_failure():
    file_hash = "a" * 32
    source_id = f"doc:{file_hash}:c-1"
    draft = {"layout": "matrix", "title": "Summary", "claims": [{"text": "x", "source_ids": [source_id]}]}
    state = {
        "agent_task_id": "task-1",
        "final_answer": "Research summary in chat.",
        "task_canvas_draft": draft,
        "task_canvas_publish_required": True,
        "task_evidence_manifest": [{"source_id": source_id}],
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


@pytest.mark.asyncio
async def test_finalizer_blocks_ungrounded_canvas_when_publish_required():
    file_hash = "a" * 32
    draft = {
        "layout": "matrix",
        "title": "Summary",
        "claims": [{"text": "Fabricated", "source_ids": [f"doc:{file_hash}:c-invented"]}],
    }
    state = {
        "agent_task_id": "task-1",
        "final_answer": "Chat summary.",
        "task_canvas_draft": draft,
        "task_canvas_publish_required": True,
        "task_evidence_manifest": [{"source_id": f"doc:{file_hash}:c-real"}],
        "task_incomplete_reasons": [],
        "task_result_warnings": [],
    }
    with patch(
        "langgraph_runtime.workflows.answer_nodes.publish_deep_research_canvas_draft",
        new=AsyncMock(side_effect=AssertionError("publish must not run")),
    ):
        result = await finalizer_node(state, {"configurable": {}})

    assert result.get("task_canvas_publish_failed", {}).get("code") == "publish_canvas_ungrounded"
