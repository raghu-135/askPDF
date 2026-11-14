"""Control-plane pytest entry: built-in graphs must pass LangGraph runtime validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from langgraph_runtime.compiler import WorkflowCompiler
from langgraph_runtime.workflows.graph_validation import GenericGraphValidator
from langgraph_runtime.workflows.node_catalog import get_node_catalog
from langgraph_runtime.workflows.validator import WorkflowValidator
from langgraph_runtime.agent.tool_registry import tool_contracts_by_id

_BUILTIN_DIR = Path(__file__).resolve().parents[1] / "workflow_catalog/builtins"
_RUNTIME_BUILTINS = (
    "auto_rag_agent",
    "router_rag_agent",
    "orchestrator_worker_rag_agent",
    "corrective_self_rag_agent",
)


def _load_spec(builtin_key: str) -> dict:
    payload = json.loads((_BUILTIN_DIR / f"{builtin_key}.json").read_text(encoding="utf-8"))
    return payload["spec_json"]


def test_node_catalog_prefetch_tools_align_with_tool_registry():
    validator = GenericGraphValidator()
    errors = validator._collect_catalog_tool_contract_errors(get_node_catalog(), tool_contracts_by_id())
    assert errors == []


@pytest.mark.parametrize("builtin_key", _RUNTIME_BUILTINS)
def test_builtin_workflow_validates_and_compiles(builtin_key: str):
    spec = _load_spec(builtin_key)
    WorkflowValidator().validate(spec)
    assert GenericGraphValidator().collect_errors(spec) == []
    WorkflowCompiler().compile(spec, checkpointer=None)