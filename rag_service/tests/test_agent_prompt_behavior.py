"""Regression tests for graph runtime prompt behavior."""

from datetime import datetime, timezone
import pytest


from app.agent.prompting import get_tool_catalog, normalize_tool_instructions, format_runtime_datetime_context


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


def test_tool_catalog_and_instruction_keys_use_canonical_retrieval_names():
    catalog = {item["tool_name"]: item for item in get_tool_catalog()}

    assert catalog["search_thread_conversation_history"]["id"] == "thread_conversation_history"
    assert catalog["search_durable_memory"]["id"] == "durable_memory"
    assert catalog["search_thread_events"]["id"] == "thread_events"

    normalized = normalize_tool_instructions(
        {
            "thread_conversation_history": "canonical instruction",
            "durable_memory": "durable instruction",
            "thread_events": "events instruction",
        }
    )

    assert normalized["thread_conversation_history"] == "canonical instruction"
    assert normalized["durable_memory"] == "durable instruction"
    assert normalized["thread_events"] == "events instruction"


def test_unknown_tool_instruction_identifier_fails_fast():
    with pytest.raises(ValueError, match="Unknown tool instruction identifiers"):
        normalize_tool_instructions({"unknown_tool": "instruction"})


def test_known_inactive_tool_instruction_is_not_mistaken_for_unknown_identifier():
    normalized = normalize_tool_instructions({"live_web_recon": "Use fresh sources"}, tool_items=["search_documents"])
    assert "live_web_recon" not in normalized
    assert "document_evidence" in normalized
