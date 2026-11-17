from __future__ import annotations

from types import SimpleNamespace

import pytest

from runtime_protocol.prompt_assembly import workflow_char_limit


def test_workflow_char_limit_uses_env_not_run_budget(monkeypatch):
    monkeypatch.setenv("DEFAULT_TOKEN_BUDGET", "10000")
    monkeypatch.setenv("CONTEXT_USABLE_RATIO", "0.80")
    monkeypatch.setenv("CONTEXT_RECENT_RATIO", "0.22")
    monkeypatch.setenv("CONTEXT_SEMANTIC_RATIO", "0.12")
    monkeypatch.setenv("CONTEXT_DURABLE_MEMORY_RATIO", "0.06")
    monkeypatch.setenv("CONTEXT_DOCUMENT_RATIO", "0.28")
    monkeypatch.setenv("CONTEXT_MEMORY_EXPANDED_RATIO", "0.12")
    monkeypatch.setenv("CONTEXT_EVIDENCE_RATIO", "0.10")
    monkeypatch.setenv("CONTEXT_DRAFT_RATIO", "0.25")
    monkeypatch.setenv("CONTEXT_PLAN_SECTION_RATIO", "0.20")
    monkeypatch.setenv("CONTEXT_PLAN_TOTAL_RATIO", "0.50")
    monkeypatch.setenv("CONTEXT_PLAN_MEMORY_RATIO", "0.12")
    env_limit = workflow_char_limit(10000, ratio_key="evidence_ratio")
    run_budget_limit = workflow_char_limit(10000, ratio_key="evidence_ratio")
    assert env_limit == run_budget_limit
    assert env_limit > 0