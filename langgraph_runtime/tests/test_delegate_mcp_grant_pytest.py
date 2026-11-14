"""LangGraph-side MCP grant and Auto builtin tool contract checks."""

from __future__ import annotations

import json
from pathlib import Path

from langgraph_runtime.api import _context
from langgraph_runtime.compiler import WorkflowCompiler
from langgraph_runtime.workflows.graph_validation import GenericGraphValidator
from langgraph_runtime.workflows.validator import WorkflowValidator
from runtime_protocol.contracts import AgentRuntimeRequest

_BUILTIN_DIR = Path(__file__).resolve().parents[2] / "rag_service/app/workflow_catalog/builtins"


def _load_spec(builtin_key: str) -> dict:
    payload = json.loads((_BUILTIN_DIR / f"{builtin_key}.json").read_text(encoding="utf-8"))
    return payload["spec_json"]


def test_execution_context_prefers_input_mcp_token_over_options():
    request = AgentRuntimeRequest(
        run_id="child-run",
        thread_id="thread-1",
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        input={"mcp_execution_context_token": "child-grant"},
        options={"mcp_execution_context_token": "parent-grant", "llm_model": "openai/gpt-4.1-mini"},
    )
    ctx = _context({"context": {"embedding_model": "embed", "resolved_spec": {"config": {}}}}, request)
    assert getattr(ctx.request, "mcp_execution_context_token") == "child-grant"
    assert ctx.agent_run_context.get("mcp_execution_context_token") == "child-grant"


def test_auto_builtin_allows_context_loader_prefetch_tools():
    auto_spec = _load_spec("auto_rag_agent")
    WorkflowValidator().validate(auto_spec)
    assert GenericGraphValidator().collect_errors(auto_spec) == []
    WorkflowCompiler().compile(auto_spec, checkpointer=None)
    allowed = set(auto_spec["config"]["allowed_tool_ids"])
    assert "document_search_knowledge" in allowed