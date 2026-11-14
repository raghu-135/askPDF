"""MCP execution grants for Auto parent runs and delegated children (control-plane)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.mcp.execution_context_token import decode_execution_context_grant
from app.product_orchestration.builtin_workflows import load_builtin_workflows
from app.runtime.adapter import RuntimeInvocationContext
from app.runtime.http_adapter import HttpLangGraphRuntimeAdapter
from runtime_protocol.contracts import AgentRuntimeRequest


def _request(*, run_id: str) -> AgentRuntimeRequest:
    return AgentRuntimeRequest(
        run_id=run_id,
        thread_id="thread-1",
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        input={"question": "What is attention?"},
        options={},
    )


@pytest.mark.asyncio
async def test_child_delegation_prepare_ignores_parent_token_in_options(monkeypatch):
    """Child prepare_request must mint a router-scoped grant even if options still hold the parent token."""

    monkeypatch.setenv("MCP_EXECUTION_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "64000")

    auto_policy = {"context_window": 8192, "use_reranker": True, "allowed_tool_ids": ["thread_shape"]}
    parent_prepared = await HttpLangGraphRuntimeAdapter("http://runtime").prepare_request(
        _request(run_id="parent-run"),
        context=RuntimeInvocationContext(embedding_model="BAAI/bge-m3", resolved_spec={"config": auto_policy}),
    )
    parent_token = parent_prepared.input["mcp_execution_context_token"]
    _, parent_allowed = decode_execution_context_grant(parent_token)
    assert "search_knowledge" not in parent_allowed

    adapter = HttpLangGraphRuntimeAdapter("http://runtime")
    router_spec = next(item["spec_json"] for item in load_builtin_workflows() if item["builtin_key"] == "router_rag_agent")
    child_request = replace(
        _request(run_id="child-run-99"),
        options={"mcp_execution_context_token": parent_token},
    )
    prepared = await adapter.prepare_request(
        child_request,
        context=RuntimeInvocationContext(embedding_model="BAAI/bge-m3", resolved_spec=router_spec),
    )
    await adapter.aclose()

    assert "mcp_execution_context_token" not in (prepared.options or {})
    child_token = prepared.input["mcp_execution_context_token"]
    child_ctx, child_allowed = decode_execution_context_grant(child_token, tool_name="search_knowledge")
    assert child_ctx.run_id == "child-run-99"
    assert "search_knowledge" in child_allowed
    assert "search_knowledge" in prepared.input["mcp_allowed_tool_ids"]


@pytest.mark.asyncio
async def test_auto_parent_grant_includes_search_knowledge(monkeypatch):
    monkeypatch.setenv("MCP_EXECUTION_CONTEXT_SECRET", "x" * 32)
    monkeypatch.setenv("HERMES_MODEL_CONTEXT_LENGTH", "64000")
    auto_spec = next(item["spec_json"] for item in load_builtin_workflows() if item["builtin_key"] == "auto_rag_agent")
    adapter = HttpLangGraphRuntimeAdapter("http://runtime")
    prepared = await adapter.prepare_request(
        _request(run_id="auto-parent"),
        context=RuntimeInvocationContext(embedding_model="BAAI/bge-m3", resolved_spec=auto_spec),
    )
    await adapter.aclose()
    _, allowed = decode_execution_context_grant(prepared.input["mcp_execution_context_token"], tool_name="search_knowledge")
    assert "search_knowledge" in allowed