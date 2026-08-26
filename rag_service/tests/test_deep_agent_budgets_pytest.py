import pytest

from app.runtime.budgets import apply_deep_agent_env_overrides, deep_agent_budgets


def test_common_configured_budgets_are_shared():
    assert deep_agent_budgets("langgraph")["max_duration_seconds"] == 7200
    assert deep_agent_budgets("hermes")["max_duration_seconds"] == 7200


def test_framework_alias_resolves_shared_budget(monkeypatch):
    monkeypatch.setenv("DEEP_AGENT_MAX_DURATION_MS", "1200000")
    monkeypatch.setenv("DEEP_AGENT_LANGGRAPH_MAX_DURATION_MS", "${DEEP_AGENT_MAX_DURATION_MS}")
    monkeypatch.setenv("DEEP_AGENT_HERMES_MAX_DURATION_MS", "${DEEP_AGENT_MAX_DURATION_MS}")
    assert deep_agent_budgets("langgraph")["max_duration_seconds"] == 1200
    assert deep_agent_budgets("hermes")["max_duration_seconds"] == 1200


def test_unsupported_framework_budget_is_not_applied(monkeypatch):
    monkeypatch.setenv("DEEP_AGENT_HERMES_DISPATCH_TIMEOUT_MS", "${DEEP_AGENT_DISPATCH_TIMEOUT_MS}")
    assert "dispatch_timeout_ms" not in deep_agent_budgets("hermes")
    resolved = apply_deep_agent_env_overrides({"dispatch_timeout_ms": 60000}, "hermes")
    assert resolved["dispatch_timeout_ms"] == 60000
    assert "worker_timeout_ms" not in resolved


def test_invalid_budget_env_fails_fast(monkeypatch):
    monkeypatch.setenv("DEEP_AGENT_MAX_EVENT_COUNT", "not-an-integer")
    monkeypatch.setenv("DEEP_AGENT_HERMES_MAX_EVENT_COUNT", "${DEEP_AGENT_MAX_EVENT_COUNT}")
    with pytest.raises(ValueError, match=r"DEEP_AGENT_(HERMES_)?MAX_EVENT_COUNT"):
        deep_agent_budgets("hermes")


def test_missing_budget_env_fails_fast(monkeypatch):
    monkeypatch.delenv("DEEP_AGENT_LANGGRAPH_MAX_EVENT_COUNT", raising=False)
    monkeypatch.delenv("DEEP_AGENT_MAX_EVENT_COUNT", raising=False)
    with pytest.raises(ValueError, match="DEEP_AGENT_LANGGRAPH_MAX_EVENT_COUNT or DEEP_AGENT_MAX_EVENT_COUNT is required"):
        deep_agent_budgets("langgraph")
