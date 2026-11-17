from __future__ import annotations

import pytest

from runtime_protocol.auto_limits import (
    AUTO_LIMIT_MAX_DELEGATES_PER_TURN,
    AUTO_LIMIT_MAX_INTERRUPTS_PER_RUN,
    AUTO_LIMIT_MAX_TOTAL_VISITS,
    auto_limit_agent_error,
    auto_limit_chat_message,
)
from langgraph_runtime.workflows.auto_limit_errors import AutoLimitExceeded, workflow_budget_as_auto_limit
from langgraph_runtime.workflows.state import WorkflowBudgetExceeded


def test_auto_limit_chat_message_mentions_thread_settings():
    message = auto_limit_chat_message(AUTO_LIMIT_MAX_DELEGATES_PER_TURN, limit=1)
    assert "AI Prompt Settings" in message
    assert "Max delegates" in message
    assert "1" in message


def test_auto_limit_agent_error_shape():
    error = auto_limit_agent_error(AUTO_LIMIT_MAX_TOTAL_VISITS, limit=12)
    assert error["code"] == "auto_limit_exceeded"
    assert error["auto_limit_kind"] == AUTO_LIMIT_MAX_TOTAL_VISITS
    assert error["limit"] == 12
    assert "safe_message" in error
    assert error["retryable"] is False


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
    exc = AutoLimitExceeded(AUTO_LIMIT_MAX_DELEGATES_PER_TURN, limit=2, state={"workflow_id": "auto_rag_agent"})
    assert "Max delegates" in exc.user_message
    assert exc.as_agent_error()["limit"] == 2