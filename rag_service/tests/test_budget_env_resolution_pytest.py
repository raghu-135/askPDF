from __future__ import annotations

import os

import pytest

from app.services.agent_task_budgets import tranche_limits, initial_budget_state
from app.runtime.budgets import deep_agent_budgets


@pytest.mark.asyncio
async def test_tranche_limits_uses_env_when_not_set_in_limits(monkeypatch):
    monkeypatch.setenv("DEEP_AGENT_MAX_MODEL_CALLS", "11")
    monkeypatch.setenv("DEEP_AGENT_MAX_MODEL_TOKENS", "2222")
    monkeypatch.setenv("DEEP_AGENT_MAX_TOOL_CALLS", "33")
    monkeypatch.setenv("DEEP_AGENT_MAX_ACTIVE_RUNTIME_MS", "4444")
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


def test_deep_agent_budgets_reads_framework_and_common_env(monkeypatch):
    # Common vars (conftest and .env.ci also set framework-specific overrides)
    for suffix, value in (
        ("MAX_MODEL_CALLS", "100"),
        ("MAX_MODEL_TOKENS", "500000"),
        ("MAX_TOOL_CALLS", "100"),
        ("MAX_ACTIVE_RUNTIME_MS", "3600000"),
    ):
        monkeypatch.setenv(f"DEEP_AGENT_{suffix}", value)
        monkeypatch.setenv(f"DEEP_AGENT_LANGGRAPH_{suffix}", value)
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
