from __future__ import annotations

import os

import pytest

from app.models.auto_limits import default_auto_limits, merge_auto_limits
from app.product_orchestration.workflow_runtime import workflow_allows_auto_limits_override
from app.product_orchestration.builtin_workflows import load_builtin_workflows


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


def test_auto_workflow_allows_auto_limits_override():
    auto = next(w for w in load_builtin_workflows() if w["builtin_key"] == "auto_rag_agent")
    spec = auto["spec_json"]
    assert workflow_allows_auto_limits_override({**spec, "workflow_id": auto["builtin_key"]})


def test_router_workflow_disallows_auto_limits_override():
    router = next(w for w in load_builtin_workflows() if w["builtin_key"] == "router_rag_agent")
    spec = router["spec_json"]
    assert not workflow_allows_auto_limits_override({**spec, "workflow_id": router["builtin_key"]})