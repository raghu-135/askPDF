from __future__ import annotations

from langgraph_runtime.workflows.auto_nodes import auto_scope_decline, ROUTER_WORKFLOW_ID
from langgraph_runtime.workflows.corrective_nodes import DOCUMENT_SCOPE_DECLINE_ANSWER
from langgraph_runtime.workflows.external_research import (
    attachment_document_scope_bypass,
    question_is_agent_runtime_meta,
    question_requests_external_research,
)
from langgraph_runtime.workflows.router_scope import router_attachment_scope_decline
from langgraph_runtime.workflows.enums import RouterRoute


def test_question_requests_external_research_detects_browser_workflow_intent():
    assert question_requests_external_research("Use the browser to find similar papers.") is True
    assert question_requests_external_research("Summarize the introduction.") is False


def test_question_is_agent_runtime_meta_detects_capability_queries():
    assert question_is_agent_runtime_meta("What tools do you have access to?") is True
    assert question_is_agent_runtime_meta("What is attention?") is False


def test_attachment_scope_bypass_for_browser_with_pdf_attached():
    state = {
        "question": "Use the browser to find similar papers.",
        "use_browser_access": True,
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_sources": [{"file_hash": "a" * 32, "score": 0.12}],
            "document_evidence_text": "",
        },
    }
    assert attachment_document_scope_bypass(state) is True
    assert auto_scope_decline(state, workflow_id=ROUTER_WORKFLOW_ID) is None
    assert router_attachment_scope_decline(state, RouterRoute.CLARIFY.value) is None
    assert router_attachment_scope_decline(state, RouterRoute.DIRECT.value) is None


def test_attachment_scope_bypass_for_meta_tool_questions():
    state = {
        "question": "What tools do you have access to?",
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_evidence_text": "",
        },
    }
    assert attachment_document_scope_bypass(state) is True
    assert router_attachment_scope_decline(state, RouterRoute.DIRECT.value) is None


def test_out_of_scope_questions_still_decline_with_attachments():
    state = {
        "question": "What is the population of Tokyo?",
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_sources": [{"file_hash": "a" * 32, "score": 0.12}],
            "document_evidence_text": "",
        },
    }
    assert attachment_document_scope_bypass(state) is False
    assert auto_scope_decline(state, workflow_id=ROUTER_WORKFLOW_ID) == DOCUMENT_SCOPE_DECLINE_ANSWER