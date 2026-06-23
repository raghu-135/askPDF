"""Regression tests for orchestrator prompt/tool exposure behavior."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent.agent import build_system_prompt


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
