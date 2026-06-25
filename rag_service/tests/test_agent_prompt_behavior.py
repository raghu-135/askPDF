"""Regression tests for orchestrator prompt/tool exposure behavior."""

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent.agent import build_system_prompt
from app.agent.agent_helpers import format_runtime_datetime_context


def test_compact_prompt_does_not_advertise_callable_tools():
    """Compact mode should not tell the model that disabled tools are callable."""
    prompt = build_system_prompt(
        context_window=8192,
        use_web_search=True,
        reasoning_mode=False,
    )

    assert "TOOL REGISTRY" not in prompt
    assert "TOOL PLAYBOOK" not in prompt
    assert "tool name: `get_thread_shape`" not in prompt
    assert "WEB SEARCH MANDATE" not in prompt
    assert "Tool calls are disabled in compact mode" in prompt


def test_reasoning_prompt_keeps_tool_registry():
    """Reasoning mode should continue to expose real bound tools."""
    prompt = build_system_prompt(
        context_window=8192,
        reasoning_mode=True,
    )

    assert "TOOL REGISTRY" in prompt
    assert "tool name: `get_thread_shape`" in prompt


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
    assert "Server current UTC datetime: 2026-06-25T19:00:00+00:00" in context


def test_system_prompt_includes_runtime_datetime_context():
    prompt = build_system_prompt(
        context_window=8192,
        client_timezone="America/Chicago",
        client_locale="en-US",
        client_now_iso="2026-06-25T19:00:00.000Z",
    )

    assert "RUNTIME DATE/TIME CONTEXT" in prompt
    assert "User timezone: America/Chicago" in prompt
