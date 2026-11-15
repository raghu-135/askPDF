"""Delegated child runs must inherit browser access from the parent execution context."""

from __future__ import annotations

from app.tools.context import ToolInvocationContext
from app.mcp.execution_context_token import issue_execution_context_token
from app.api.internal_agent_runs import DelegateSyncRequest, _validate_execution_context


def test_delegate_request_accepts_browser_fields():
    body = DelegateSyncRequest(
        parent_run_id="parent",
        thread_id="thread-1",
        child_workflow_id="router_rag_agent",
        question="Use the browser to find similar papers.",
        use_browser_access=True,
        browser_access_mode="ask",
    )
    assert body.use_browser_access is True
    assert body.browser_access_mode == "ask"


def test_execution_context_validation_exposes_browser_use(monkeypatch):
    monkeypatch.setenv("MCP_EXECUTION_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "64000")
    token = issue_execution_context_token(
        ToolInvocationContext(
            thread_id="thread-1",
            run_id="parent",
            embedding_model="BAAI/bge-m3",
            use_browser_access=True,
            use_web_search=False,
        ),
        task_id="parent",
        allowed_tools=["browser_navigate"],
        ttl_seconds=120,
        runtime="langgraph",
    )
    fields = _validate_execution_context(token, parent_run_id="parent", thread_id="thread-1")
    assert fields["use_browser_access"] is True