"""Regression tests for orchestrator prompt/tool exposure behavior."""

import inspect
from datetime import datetime, timezone


from app.agent.agent import build_system_prompt, _format_prefetch_for_prompt
from app.agent.agent_helpers import format_runtime_datetime_context


def test_system_prompt_has_no_reasoning_mode_argument():
    """Non-reasoning orchestration support should not be part of the prompt API."""
    signature = inspect.signature(build_system_prompt)

    assert "reasoning_mode" not in signature.parameters


def test_system_prompt_exposes_tool_registry():
    """The orchestrator prompt should expose real bound tools."""
    prompt = build_system_prompt(
        context_window=8192,
    )

    assert "TOOL REGISTRY" in prompt
    assert "tool name: `get_thread_shape`" in prompt
    assert "tool name: `search_thread_timeline`" in prompt
    assert "find_topic_anchor_in_history" not in prompt


def test_system_prompt_includes_web_search_mandate_when_enabled():
    prompt = build_system_prompt(
        context_window=8192,
        use_web_search=True,
    )

    assert "WEB SEARCH MANDATE" in prompt
    assert "Tool calls are disabled in compact mode" not in prompt


def test_runtime_datetime_context_uses_browser_timezone_with_server_clock():
    """Runtime clock should be computed in the browser timezone from server UTC."""
    context = format_runtime_datetime_context(
        client_timezone="America/Chicago",
        client_locale="en-US",
        client_now_iso="2026-06-25T19:00:00.000Z",
        now_utc=datetime(2026, 6, 25, 19, 0, tzinfo=timezone.utc),
    )

    assert "RUNTIME DATE/TIME CONTEXT" in context
    assert "User timezone: America/Chicago" in context
    assert "User locale: en-US" in context
    assert "User-local current datetime: 2026-06-25T14:00:00-05:00" in context
    assert "Server current UTC datetime: 2026-06-25T19:00:00Z" in context


def test_system_prompt_includes_runtime_datetime_context():
    prompt = build_system_prompt(
        context_window=8192,
        client_timezone="America/Chicago",
        client_locale="en-US",
        client_now_iso="2026-06-25T19:00:00.000Z",
    )

    assert "RUNTIME DATE/TIME CONTEXT" in prompt
    assert "User timezone: America/Chicago" in prompt


def test_system_prompt_includes_temporal_metadata_contract():
    prompt = build_system_prompt(context_window=8192)

    assert "TEMPORAL METADATA CONTRACT" in prompt
    assert "message_created_at" in prompt
    assert "document_available_in_thread_at" in prompt
    assert "web_search_performed_at" in prompt
    assert "timeline_event_at" in prompt
    assert "search_thread_timeline" in prompt


def test_prefetch_document_inventory_includes_document_level_counts():
    prompt = _format_prefetch_for_prompt(
        {
            "documents": [
                {
                    "index": 1,
                    "file_name": "report.pdf",
                    "file_hash": "file-1",
                    "source_type": "pdf",
                    "document_available_in_thread_at": "2026-06-26T00:00:00Z",
                    "page_count": 4,
                    "word_count": 123,
                    "sentence_count": 12,
                    "chunk_count": 5,
                }
            ]
        }
    )

    assert "pages: 4" in prompt
    assert "words: 123" in prompt
    assert "sentences: 12" in prompt
    assert "chunks: 5" in prompt
