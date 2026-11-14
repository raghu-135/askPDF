"""Auto orchestrator workflow contract tests."""

from __future__ import annotations

import pytest

from langgraph_runtime.compiler import WorkflowCompiler
from langgraph_runtime.workflows.auto_nodes import (
    DEEP_RESEARCH_WORKFLOW_ID,
    ROUTER_WORKFLOW_ID,
    auto_scope_decline,
)
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


def test_auto_scope_decline_for_router_without_prefetch_evidence():
    state = {
        "question": "What is the kinetic energy of a satellite in low Earth orbit?",
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_evidence_text": "",
        },
    }
    assert auto_scope_decline(state, workflow_id=ROUTER_WORKFLOW_ID) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_auto_scope_decline_allows_router_when_prefetch_has_evidence():
    state = {
        "question": "Summarize the introduction.",
        "pre_fetch_bundle": {
            "document_sources": [{"file_hash": "a" * 32, "score": 0.88}],
            "document_evidence_text": "Attention is all you need.",
        },
    }
    assert auto_scope_decline(state, workflow_id=ROUTER_WORKFLOW_ID) is None


def test_auto_scope_decline_for_router_when_prefetch_scores_are_low():
    state = {
        "question": "What regulatory capital ratio applies under Basel IV in 2030?",
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_sources": [{"file_hash": "a" * 32, "score": 0.21}],
            "document_evidence_text": "Table 2 training cost for the base model.",
        },
    }
    assert auto_scope_decline(state, workflow_id=ROUTER_WORKFLOW_ID) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_auto_scope_decline_skips_deep_research_delegate():
    state = {
        "question": "Please write a literature review of transformer papers.",
        "pre_fetch_bundle": {
            "document_sources": [{"file_hash": "a" * 32}],
            "document_evidence_text": "",
        },
    }
    assert auto_scope_decline(state, workflow_id=DEEP_RESEARCH_WORKFLOW_ID) is None
