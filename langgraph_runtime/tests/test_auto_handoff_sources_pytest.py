"""Regression tests for Auto delegate source handoff."""

from __future__ import annotations

import asyncio

import pytest

from langgraph_runtime.workflows.auto_handoff_sources import merge_auto_handoff_sources
from langgraph_runtime.workflows.auto_nodes import delegate_synthesizer_node


def test_merge_auto_handoff_sources_dedupes_parent_and_child_rows():
    state = {
        "document_sources": [{"file_hash": "abc", "page_number": 1, "snippet": "prefetch"}],
        "web_sources": [],
        "used_chat_ids": ["chat-a"],
        "pre_fetch_bundle": {
            "document_sources": [{"file_hash": "abc", "page_number": 1, "snippet": "prefetch duplicate"}],
            "web_sources": [{"url": "https://example.com/doc", "title": "Example"}],
            "used_chat_ids": ["chat-a"],
        },
        "delegate_child_output": {
            "result": {
                "document_sources": [{"file_hash": "abc", "page_number": 2, "snippet": "child page"}],
                "web_sources": [{"url": "https://example.com/doc", "title": "Example child"}],
                "used_chat_ids": ["chat-b"],
            },
        },
    }
    merged = merge_auto_handoff_sources(state)
    assert len(merged["document_sources"]) == 2
    assert merged["web_sources"] == [{"url": "https://example.com/doc", "title": "Example child"}]
    assert merged["used_chat_ids"] == ["chat-a", "chat-b"]


@pytest.mark.asyncio
async def test_delegate_synthesizer_node_merges_child_sources_into_state():
    state = {
        "document_sources": [],
        "web_sources": [],
        "used_chat_ids": [],
        "pre_fetch_bundle": {"document_sources": [{"file_hash": "f-1", "snippet": "from prefetch"}]},
        "child_final_answer": "Done.",
        "child_run_id": "child-run-1",
        "child_workflow_id": "router_rag_agent",
        "delegate_child_output": {
            "result": {
                "document_sources": [{"file_hash": "f-1", "page_number": 3, "snippet": "from child"}],
                "web_sources": [{"url": "https://arxiv.org/abs/1706.03762"}],
                "used_chat_ids": ["turn-9"],
            },
        },
        "node_events": [],
        "debug_trace": {},
    }

    update = await delegate_synthesizer_node(state, {})
    assert update["final_answer"] == "Done."
    assert len(update["document_sources"]) == 1
    assert update["document_sources"][0]["page_number"] == 3
    assert update["web_sources"][0]["url"] == "https://arxiv.org/abs/1706.03762"
    assert update["used_chat_ids"] == ["turn-9"]
    assert update["debug_trace"]["technical_details"]["child_run_id"] == "child-run-1"