from __future__ import annotations

from runtime_protocol.auto_limits import AUTO_LIMIT_MAX_TOTAL_VISITS
from langgraph_runtime.workflows.auto_limit_errors import AutoLimitExceeded, workflow_budget_as_auto_limit
from langgraph_runtime.workflows.state import WorkflowBudgetExceeded


def test_workflow_budget_maps_to_auto_limit_for_auto_workflow():
    exc = WorkflowBudgetExceeded(
        limit=6,
        node_id="auto_orchestrator",
        node_type="auto_orchestrator",
        visit_index=2,
        observed_visits=6,
    )
    state = {"workflow_id": "auto_rag_agent", "loop_policy": {"max_total_visits": 6}}
    mapped = workflow_budget_as_auto_limit(exc, state)
    assert isinstance(mapped, AutoLimitExceeded)
    assert mapped.kind == AUTO_LIMIT_MAX_TOTAL_VISITS
    assert "Max visits" in mapped.user_message


def test_auto_limit_exceeded_user_message():
    exc = AutoLimitExceeded("max_delegates_per_turn", limit=2, state={"workflow_id": "auto_rag_agent"})
    assert "Max delegates" in exc.user_message
    assert exc.as_agent_error()["limit"] == 2