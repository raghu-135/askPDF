import pytest

from app.mcp.execution_context_token import (
    decode_execution_context_token,
    issue_execution_context_token,
)
from app.tools.context import ToolInvocationContext


def test_signed_context_round_trip_and_tool_allowlist(monkeypatch):
    monkeypatch.setenv("HERMES_MCP_CONTEXT_SECRET", "x" * 32)
    token = issue_execution_context_token(
        ToolInvocationContext(thread_id="thread-1", run_id="run-1", embedding_model="embed", context_window=8192),
        task_id="task-1",
        allowed_tools=["search_documents"],
    )
    decoded = decode_execution_context_token(token, tool_name="search_documents")
    assert decoded.thread_id == "thread-1"
    assert decoded.run_id == "run-1"
    assert decoded.context_window == 8192
    assert decoded.extensions["task_id"] == "task-1"
    with pytest.raises(ValueError, match="Invalid Hermes MCP"):
        decode_execution_context_token(token, tool_name="search_thread_events")


def test_signed_context_rejects_tampering(monkeypatch):
    monkeypatch.setenv("HERMES_MCP_CONTEXT_SECRET", "x" * 32)
    token = issue_execution_context_token(
        ToolInvocationContext(thread_id="thread-1"),
        task_id="task-1",
        allowed_tools=["search_documents"],
    )
    with pytest.raises(ValueError, match="Invalid Hermes MCP"):
        decode_execution_context_token(token + "tampered", tool_name="search_documents")


def test_signed_context_rejects_expiry_and_incomplete_identity(monkeypatch):
    from app.mcp import execution_context_token as token_module

    monkeypatch.setenv("HERMES_MCP_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setattr(token_module.time, "time", lambda: 100)
    token = issue_execution_context_token(
        ToolInvocationContext(thread_id="thread-1", run_id="run-1", context_window=8192),
        task_id="task-1",
        allowed_tools=["search_documents"],
        ttl_seconds=60,
    )
    monkeypatch.setattr(token_module.time, "time", lambda: 200)
    with pytest.raises(ValueError, match="Invalid Hermes MCP"):
        decode_execution_context_token(token, tool_name="search_documents")

    monkeypatch.setattr(token_module.time, "time", lambda: 100)
    incomplete = issue_execution_context_token(
        ToolInvocationContext(thread_id="thread-1", context_window=8192),
        task_id="task-1",
        allowed_tools=["search_documents"],
    )
    with pytest.raises(ValueError, match="Invalid Hermes MCP"):
        decode_execution_context_token(incomplete, tool_name="search_documents")
