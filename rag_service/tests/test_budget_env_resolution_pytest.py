from __future__ import annotations

import os

import pytest

from app.services.agent_task_budgets import tranche_limits, initial_budget_state
from app.runtime.budgets import deep_agent_budgets


def _set_required_env():
    os.environ["DEEP_AGENT_MAX_MODEL_CALLS"] = "11"
    os.environ["DEEP_AGENT_MAX_MODEL_TOKENS"] = "2222"
    os.environ["DEEP_AGENT_MAX_TOOL_CALLS"] = "33"
    os.environ["DEEP_AGENT_MAX_ACTIVE_RUNTIME_MS"] = "4444"


def _clear_required_env():
    for key in (
        "DEEP_AGENT_MAX_MODEL_CALLS",
        "DEEP_AGENT_MAX_MODEL_TOKENS",
        "DEEP_AGENT_MAX_TOOL_CALLS",
        "DEEP_AGENT_MAX_ACTIVE_RUNTIME_MS",
    ):
        os.environ.pop(key, None)


@pytest.mark.asyncio
async def test_tranche_limits_uses_env_when_not_set_in_limits(monkeypatch):
    try:
        _set_required_env()
        # tranche_limits requires all four keys via either env or limits
        limits = tranche_limits({})
        assert limits == {
            "model_calls": 11,
            "model_tokens": 2222,
            "tool_calls": 33,
            "elapsed_active_ms": 4444,
        }
        # initial_budget_state wires the tranche limits into the envelope
        state = initial_budget_state({})
        assert state["tranche_limits"] == limits
    finally:
        _clear_required_env()


def test_deep_agent_budgets_reads_framework_and_common_env(monkeypatch):
    # Common vars
    monkeypatch.setenv("DEEP_AGENT_MAX_MODEL_CALLS", "100")
    monkeypatch.setenv("DEEP_AGENT_MAX_MODEL_TOKENS", "500000")
    monkeypatch.setenv("DEEP_AGENT_MAX_TOOL_CALLS", "100")
    monkeypatch.setenv("DEEP_AGENT_MAX_ACTIVE_RUNTIME_MS", "3600000")
    # LangGraph-specific timeouts
    monkeypatch.setenv("DEEP_AGENT_LANGGRAPH_DISPATCH_TIMEOUT_MS", "600000")
    monkeypatch.setenv("DEEP_AGENT_LANGGRAPH_WORKER_TIMEOUT_MS", "600000")
    monkeypatch.setenv("DEEP_AGENT_LANGGRAPH_WEB_WORKER_TIMEOUT_MS", "600000")
    monkeypatch.setenv("DEEP_AGENT_LANGGRAPH_SUBAGENT_TIMEOUT_MS", "600000")
    budgets = deep_agent_budgets("langgraph")
    assert budgets["max_model_calls"] == 100
    assert budgets["max_model_tokens"] == 500000
    assert budgets["max_tool_calls"] == 100
    assert budgets["max_active_runtime_ms"] == 3600000
    assert budgets["dispatch_timeout_ms"] == 600000
    assert budgets["worker_timeout_ms"] == 600000
    assert budgets["web_worker_timeout_ms"] == 600000
