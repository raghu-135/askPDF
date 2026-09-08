"""Regression tests for graph runtime prompt behavior."""

from datetime import datetime, timezone


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


def test_tool_catalog_and_legacy_instruction_keys_use_canonical_retrieval_names():
    catalog = {item["tool_name"]: item for item in get_tool_catalog()}

    assert catalog["search_thread_conversation_history"]["id"] == "thread_conversation_history"
    assert catalog["search_durable_memory"]["id"] == "durable_memory"
    assert catalog["search_thread_events"]["id"] == "thread_events"

    normalized = normalize_tool_instructions(
        {
            "deep_memory": "legacy instruction",
            "thread_conversation_history": "canonical instruction",
            "memory_recall": "durable instruction",
            "thread_timeline": "events instruction",
        }
    )

    assert normalized["thread_conversation_history"] == "canonical instruction"
    assert normalized["durable_memory"] == "durable instruction"
    assert normalized["thread_events"] == "events instruction"
