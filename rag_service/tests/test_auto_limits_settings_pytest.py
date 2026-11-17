from __future__ import annotations

import os

import pytest

from app.models.auto_limits import default_auto_limits, merge_auto_limits


@pytest.fixture(autouse=True)
def _auto_env(monkeypatch):
    monkeypatch.setenv("AUTO_MAX_TOTAL_VISITS", "12")
    monkeypatch.setenv("AUTO_MAX_INTERRUPTS_PER_RUN", "4")
    monkeypatch.setenv("AUTO_MAX_DELEGATES_PER_TURN", "1")
    monkeypatch.setenv("AUTO_REQUIRE_PLAN_APPROVAL_ON_COMPLEX", "false")
    monkeypatch.setenv("AUTO_REQUIRE_PLAN_APPROVAL_ALL", "false")


def test_default_auto_limits_from_env():
    defaults = default_auto_limits()
    assert defaults["max_total_visits"] == 12
    assert defaults["max_delegates_per_turn"] == 1


def test_merge_auto_limits_clamps():
    merged = merge_auto_limits({"max_delegates_per_turn": 99, "max_total_visits": 4})
    assert merged["max_delegates_per_turn"] == 3
    assert merged["max_total_visits"] == 4


def test_missing_auto_env_raises(monkeypatch):
    monkeypatch.delenv("AUTO_MAX_TOTAL_VISITS", raising=False)
    with pytest.raises(ValueError, match="AUTO_MAX_TOTAL_VISITS"):
        default_auto_limits()