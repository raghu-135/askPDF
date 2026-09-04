from __future__ import annotations

import hashlib

import pytest

from langgraph_runtime.dependencies import DependencyMonitor
from langgraph_runtime.workflows.deep_research_execution import RuntimeBudgetMeter, RuntimeExecutionServices
from langgraph_runtime.workflows.planning import normalize_execution_plan
from app.runtime.http_adapter import HttpLangGraphRuntimeAdapter
from runtime_protocol.errors import RuntimeError as AgentRuntimeError


def _budget(limit: int = 10) -> dict:
    zero = {"model_calls": 0, "model_tokens": 0, "tool_calls": 0, "elapsed_active_ms": 0}
    return {
        "tranche_index": 1,
        "tranche_limits": {"model_calls": limit, "model_tokens": limit * 10, "tool_calls": limit, "elapsed_active_ms": limit * 1000},
        "tranche_usage": dict(zero),
        "lifetime_usage": dict(zero),
    }


def test_budget_meter_uses_snapshot_and_rejects_missing_snapshot():
    with pytest.raises(AgentRuntimeError, match="budget snapshot"):
        RuntimeBudgetMeter({}, {"max_model_calls": 100})
    meter = RuntimeBudgetMeter(_budget(2), {"max_model_calls": 2})
    with pytest.raises(AgentRuntimeError, match="contradictory"):
        RuntimeBudgetMeter(_budget(2), {"max_model_calls": 100})
    assert meter._limits["model_calls"] == 2


@pytest.mark.asyncio
async def test_artifact_reports_require_matching_content_and_digest():
    content = "inherited evidence"
    services = RuntimeExecutionServices(
        todos=None, artifacts=None, budgets=None, cancellation=type("C", (), {"requested": lambda self: False})(),
        events=None, memory=None, state={"runtime_artifacts": []},
    )
    manifest = {"id": "a1", "sha256": hashlib.sha256(content.encode()).hexdigest(), "byte_size": len(content.encode())}
    with pytest.raises(AgentRuntimeError, match="unavailable"):
        await services.report_contents({"task_evidence_manifest": [manifest], "task_evidence_gaps": []})

    services.state = {"runtime_artifacts": [{"id": "a1", "content": content}]}
    reports, _ = await services.report_contents({"task_evidence_manifest": [manifest], "task_evidence_gaps": []})
    assert reports == [content]

    bad = {**manifest, "sha256": "bad"}
    with pytest.raises(AgentRuntimeError, match="integrity"):
        await services.report_contents({"task_evidence_manifest": [bad], "task_evidence_gaps": []})


def test_planner_does_not_fabricate_clarification_choices():
    with pytest.raises(AgentRuntimeError, match="clarification choices"):
        normalize_execution_plan({"route": "clarify", "clarification_options": []}, use_web_search=False)


def test_dependency_monitor_accepts_zero_jitter(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_DEPENDENCY_JITTER_RATIO", "0")
    monitor = DependencyMonitor()
    assert monitor.jitter == 0


def test_langgraph_connector_requires_authentication(monkeypatch):
    monkeypatch.delenv("LANGGRAPH_RUNTIME_TOKEN", raising=False)
    with pytest.raises(AgentRuntimeError, match="TOKEN"):
        HttpLangGraphRuntimeAdapter("http://runtime")
