import pytest

from app.runtime.budgets import apply_deep_agent_env_overrides, deep_agent_budgets


def test_common_defaults_are_shared():
    assert deep_agent_budgets("langgraph")["max_duration_seconds"] == 3600
    assert deep_agent_budgets("hermes")["max_duration_seconds"] == 3600


def test_framework_override_wins_over_common(monkeypatch):
    monkeypatch.setenv("DEEP_AGENT_MAX_DURATION_MS", "1200000")
    monkeypatch.setenv("DEEP_AGENT_HERMES_MAX_DURATION_MS", "900000")
    assert deep_agent_budgets("langgraph")["max_duration_seconds"] == 1200
    assert deep_agent_budgets("hermes")["max_duration_seconds"] == 900


def test_unsupported_framework_budget_is_not_applied(monkeypatch):
    monkeypatch.setenv("DEEP_AGENT_HERMES_DISPATCH_TIMEOUT_MS", "90000")
    assert "dispatch_timeout_ms" not in deep_agent_budgets("hermes")
    assert apply_deep_agent_env_overrides({"dispatch_timeout_ms": 60000}, "hermes") == {
        "dispatch_timeout_ms": 60000,
    }


def test_invalid_budget_env_fails_fast(monkeypatch):
    monkeypatch.setenv("DEEP_AGENT_MAX_EVENT_COUNT", "not-an-integer")
    with pytest.raises(ValueError, match="DEEP_AGENT_MAX_EVENT_COUNT"):
        deep_agent_budgets("hermes")
