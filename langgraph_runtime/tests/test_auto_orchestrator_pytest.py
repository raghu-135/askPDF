"""Auto orchestrator workflow contract tests."""

from __future__ import annotations

import pytest

from langgraph_runtime.compiler import WorkflowCompiler
from langgraph_runtime.workflows.auto_nodes import auto_scope_decline
from langgraph_runtime.workflows.corrective_nodes import DOCUMENT_SCOPE_DECLINE_ANSWER
from langgraph_runtime.workflows.graph_validation import WorkflowGraphValidator
import json
from pathlib import Path


def _auto_spec() -> dict:
    path = Path(__file__).resolve().parents[2] / "rag_service/app/workflow_catalog/builtins/auto_rag_agent.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["spec_json"]


def test_auto_builtin_validates_and_compiles():
    spec = _auto_spec()
    errors = WorkflowGraphValidator().validate(spec)
    assert errors == []
    WorkflowCompiler().compile(spec, checkpointer=None)


def test_auto_scope_decline_for_world_fact_without_evidence():
    state = {
        "question": "What is the population of Tokyo?",
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_evidence_text": "",
        },
    }
    assert auto_scope_decline(state) == DOCUMENT_SCOPE_DECLINE_ANSWER
