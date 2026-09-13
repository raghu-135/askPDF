import pytest

from app.mcp.execution_context_token import (
    ExecutionContextTokenError,
    decode_execution_context_token,
    issue_execution_context_token,
    validate_execution_context_identity,
)
from app.tools.context import ToolInvocationContext


def test_signed_context_round_trip_and_tool_allowlist(monkeypatch):
    monkeypatch.setenv("MCP_EXECUTION_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "8192")
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
    with pytest.raises(ValueError, match="Invalid MCP"):
        decode_execution_context_token(token, tool_name="search_thread_events")


def test_signed_context_rejects_tampering(monkeypatch):
    monkeypatch.setenv("MCP_EXECUTION_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "8192")
    token = issue_execution_context_token(
        ToolInvocationContext(thread_id="thread-1", run_id="run-1"),
        task_id="task-1",
        allowed_tools=["search_documents"],
    )
    with pytest.raises(ExecutionContextTokenError, match="Invalid MCP") as rejected:
        decode_execution_context_token(token + "tampered", tool_name="search_documents")
    assert rejected.value.reason == "bad_signature"


def test_signed_context_rejects_wrong_audience(monkeypatch):
    from app.mcp import execution_context_token as token_module

    monkeypatch.setenv("MCP_EXECUTION_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "8192")
    monkeypatch.setattr(token_module, "TOKEN_AUDIENCE", "wrong-service")
    token = issue_execution_context_token(
        ToolInvocationContext(thread_id="thread-1", run_id="run-1", context_window=8192),
        task_id="task-1",
        allowed_tools=["search_documents"],
    )
    monkeypatch.setattr(token_module, "TOKEN_AUDIENCE", "askpdf-mcp")
    with pytest.raises(ExecutionContextTokenError) as rejected:
        decode_execution_context_token(token)
    assert rejected.value.reason == "wrong_audience"


def test_signed_context_ttl_has_no_unconditional_hour_minimum(monkeypatch):
    from app.mcp import execution_context_token as token_module

    monkeypatch.setenv("MCP_EXECUTION_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "8192")
    monkeypatch.setattr(token_module.time, "time", lambda: 100)
    token = issue_execution_context_token(
        ToolInvocationContext(thread_id="thread-1", run_id="run-1", context_window=8192),
        task_id="task-1",
        allowed_tools=["search_documents"],
        ttl_seconds=1,
    )
    monkeypatch.setattr(token_module.time, "time", lambda: 102)
    with pytest.raises(ExecutionContextTokenError) as rejected:
        decode_execution_context_token(token)
    assert rejected.value.reason == "expired"


def test_signed_context_rejects_expiry_and_incomplete_identity(monkeypatch):
    from app.mcp import execution_context_token as token_module

    monkeypatch.setenv("MCP_EXECUTION_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "8192")
    monkeypatch.setattr(token_module.time, "time", lambda: 100)
    token = issue_execution_context_token(
        ToolInvocationContext(thread_id="thread-1", run_id="run-1", context_window=8192),
        task_id="task-1",
        allowed_tools=["search_documents"],
        ttl_seconds=60,
    )
    monkeypatch.setattr(token_module.time, "time", lambda: 200)
    with pytest.raises(ExecutionContextTokenError, match="Invalid MCP") as expired:
        decode_execution_context_token(token, tool_name="search_documents")
    assert expired.value.reason == "expired"

    monkeypatch.setattr(token_module.time, "time", lambda: 100)
    with pytest.raises(ValueError, match="requires thread, run, and task identities"):
        issue_execution_context_token(
            ToolInvocationContext(thread_id="thread-1", context_window=8192),
            task_id="task-1",
            allowed_tools=["search_documents"],
        )


def test_signed_context_rejects_deployment_context_mismatch(monkeypatch):
    monkeypatch.setenv("MCP_EXECUTION_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "8192")
    token = issue_execution_context_token(
        ToolInvocationContext(thread_id="thread-1", run_id="run-1", context_window=8192),
        task_id="task-1",
        allowed_tools=["search_documents"],
    )
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "32768")
    with pytest.raises(ExecutionContextTokenError) as rejected:
        decode_execution_context_token(token, tool_name="search_documents")
    assert rejected.value.reason == "model_context_mismatch"


def test_langgraph_context_window_is_not_checked_against_hermes_limit(monkeypatch):
    monkeypatch.setenv("MCP_EXECUTION_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "32768")
    token = issue_execution_context_token(
        ToolInvocationContext(thread_id="thread-1", run_id="run-1", context_window=8192),
        task_id="task-1",
        allowed_tools=["search_documents"],
        runtime="langgraph",
    )

    decoded = decode_execution_context_token(token, tool_name="search_documents")

    assert decoded.context_window == 8192


def test_execution_context_identity_rejects_cross_run_reuse(monkeypatch):
    monkeypatch.setenv("MCP_EXECUTION_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "8192")
    token = issue_execution_context_token(
        ToolInvocationContext(thread_id="thread-1", run_id="run-1", context_window=8192),
        task_id="task-1",
        allowed_tools=["search_documents"],
    )
    context = decode_execution_context_token(token)
    validate_execution_context_identity(context, run_id="run-1", thread_id="thread-1", task_id="task-1")
    with pytest.raises(ExecutionContextTokenError) as rejected:
        validate_execution_context_identity(context, run_id="run-2", thread_id="thread-1", task_id="task-1")
    assert rejected.value.reason == "identity_mismatch"
