from __future__ import annotations

from langgraph_runtime.workflows.document_scope import (
    DOCUMENT_SCOPE_DECLINE_ANSWER,
    attachment_retrieval_scope_decline,
    document_retrieval_scores,
)
from langgraph_runtime.workflows.router_scope import router_attachment_scope_decline
from langgraph_runtime.workflows.enums import RouterRoute


def test_document_retrieval_scores_merge_prefetch_and_packet_sources():
    state = {
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_sources": [{"file_hash": "a" * 32, "score": 0.22}],
        },
        "evidence_packets": [
            {
                "source_ids": [f"doc:{'a' * 32}:chunk-1"],
                "raw_retriever_score": 0.31,
            }
        ],
    }
    assert document_retrieval_scores(state) == [0.22, 0.31]


def test_attachment_scope_declines_on_low_prefetch_scores_despite_evidence_text():
    state = {
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_evidence_text": "Unrelated PDF excerpt about model architecture.",
            "document_sources": [{"file_hash": "a" * 32, "score": 0.18}],
        },
        "question": "What is the population of Tokyo?",
    }
    assert attachment_retrieval_scope_decline(state) == DOCUMENT_SCOPE_DECLINE_ANSWER
    assert router_attachment_scope_decline(state, RouterRoute.CLARIFY.value) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_attachment_scope_keeps_in_scope_high_scoring_prefetch():
    state = {
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_evidence_text": "Table 2 lists training cost 3.3 x 10^18 FLOPs.",
            "document_sources": [{"file_hash": "a" * 32, "score": 0.81}],
        },
        "question": "What is the EN-FR training cost in Table 2?",
    }
    assert attachment_retrieval_scope_decline(state) is None
    assert router_attachment_scope_decline(state, RouterRoute.CLARIFY.value) is None
