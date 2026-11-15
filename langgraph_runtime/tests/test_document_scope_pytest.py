from __future__ import annotations

from langgraph_runtime.workflows.document_scope import (
    DOCUMENT_SCOPE_DECLINE_ANSWER,
    attachment_retrieval_scope_decline,
    document_retrieval_scores,
    postprocess_grounded_document_answer,
    workflow_post_retrieval_scope_decline,
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


def test_tokyo_declines_via_low_prefetch_scores_not_lexical_matching():
    state = {
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_evidence_text": "Attention Is All You Need discusses transformer layers.",
            "document_sources": [{"file_hash": "a" * 32, "score": 0.55}],
        },
        "question": "What is the population of Tokyo?",
        "workflow_id": "router_rag_agent",
        "route": "document",
    }
    # High prefetch score alone does not decline; router proceeds to synthesis.
    assert workflow_post_retrieval_scope_decline(state) is None
    state_low_score = {
        **state,
        "pre_fetch_bundle": {
            **state["pre_fetch_bundle"],
            "document_sources": [{"file_hash": "a" * 32, "score": 0.18}],
        },
    }
    assert workflow_post_retrieval_scope_decline(state_low_score) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_natural_language_questions_not_declined_when_retrieval_is_relevant():
    base_bundle = {
        "documents": [{"file_hash": "a" * 32}],
        "document_evidence_text": (
            "We present results on WMT translation. Our method uses stacked attention. "
            "Authors include Ashish Vaswani and colleagues."
        ),
        "document_sources": [{"file_hash": "a" * 32, "score": 0.72}],
    }
    for question in (
        "Summarize the results.",
        "Explain the method.",
        "Who are the authors?",
    ):
        state = {
            "pre_fetch_bundle": base_bundle,
            "question": question,
            "workflow_id": "router_rag_agent",
        }
        assert attachment_retrieval_scope_decline(state) is None
        assert workflow_post_retrieval_scope_decline(state) is None


def test_router_post_retrieval_decline_not_blocked_by_document_route():
    state = {
        "pre_fetch_bundle": {"documents": [{"file_hash": "a" * 32}]},
        "evidence_packets": [
            {
                "source_ids": [f"doc:{'a' * 32}:chunk-1"],
                "content": "",
                "raw_retriever_score": 0.21,
            }
        ],
        "document_sources": [{"file_hash": "a" * 32, "score": 0.21}],
        "question": "What is the population of Tokyo?",
        "workflow_id": "router_rag_agent",
        "route": "document",
    }
    assert workflow_post_retrieval_scope_decline(state) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_router_post_retrieval_keeps_in_scope_when_doc_packets_lack_scores():
    state = {
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_sources": [{"file_hash": "a" * 32, "score": 0.12}],
        },
        "evidence_packets": [
            {
                "source_ids": [f"doc:{'a' * 32}:chunk-1"],
                "content": "Table 2 big-model EN-FR training cost is 2.3 x 10^19 FLOPs.",
            }
        ],
        "question": "For the Transformer (big model) in Table 2, what is the EN-FR training cost in FLOPs?",
        "workflow_id": "router_rag_agent",
        "route": "document",
    }
    assert workflow_post_retrieval_scope_decline(state) is None


def test_attachment_scope_keeps_in_scope_when_packets_have_content_despite_low_scores():
    state = {
        "pre_fetch_bundle": {"documents": [{"file_hash": "a" * 32}]},
        "evidence_packets": [
            {
                "source_ids": [f"doc:{'a' * 32}:chunk-1"],
                "content": "Table 2 big-model EN-FR training cost is 2.3 x 10^19 FLOPs.",
                "raw_retriever_score": 0.21,
            }
        ],
        "question": "For the Transformer (big model) in Table 2, what is the EN-FR training cost in FLOPs?",
    }
    assert attachment_retrieval_scope_decline(state) is None


def test_router_post_retrieval_keeps_in_scope_on_high_worker_score_despite_low_prefetch():
    state = {
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_sources": [{"file_hash": "a" * 32, "score": 0.18}],
        },
        "evidence_packets": [
            {
                "source_ids": [f"doc:{'a' * 32}:chunk-1"],
                "content": "BLEU EN-DE score is 28.4.",
                "raw_retriever_score": 0.91,
            }
        ],
        "question": "What is the EN-DE BLEU score in Table 2?",
        "workflow_id": "router_rag_agent",
        "route": "document",
    }
    assert workflow_post_retrieval_scope_decline(state) is None


def test_attachment_scope_keeps_short_encoder_in_scope_on_weak_worker_scores():
    state = {
        "pre_fetch_bundle": {"documents": [{"file_hash": "a" * 32}]},
        "evidence_packets": [
            {
                "source_ids": [f"doc:{'a' * 32}:chunk-1"],
                "content": "",
                "raw_retriever_score": 0.35,
            }
        ],
        "question": "How many encoder layers does the base Transformer use? Reply with the number only.",
    }
    assert attachment_retrieval_scope_decline(state) is None


def test_relevance_decline_skips_low_confidence_irrelevant_assessments():
    from langgraph_runtime.workflows.document_scope import decline_when_all_document_packets_irrelevant

    packet_id = "retrieval_worker:1"
    state = {
        "corrective_policy": {"minimum_relevance_confidence": 0.65},
        "evidence_packets": [
            {
                "id": packet_id,
                "source_ids": [f"doc:{'a' * 32}:chunk-1"],
                "content": "The encoder stack uses N = 6 identical layers.",
                "raw_retriever_score": 0.38,
            }
        ],
        "evidence_assessments": [
            {
                "packet_id": packet_id,
                "relevant": False,
                "confidence": 0.2,
            }
        ],
    }
    assert decline_when_all_document_packets_irrelevant(state) is None


def test_relevance_decline_skips_when_weak_scores_but_packets_have_content():
    from langgraph_runtime.workflows.document_scope import decline_when_all_document_packets_irrelevant

    packet_id = "retrieval_worker:1"
    state = {
        "corrective_policy": {"minimum_relevance_confidence": 0.65},
        "evidence_packets": [
            {
                "id": packet_id,
                "source_ids": [f"doc:{'a' * 32}:chunk-1"],
                "content": "We propose the Transformer architecture.",
                "raw_retriever_score": 0.36,
            }
        ],
        "evidence_assessments": [
            {
                "packet_id": packet_id,
                "relevant": False,
                "confidence": 0.95,
            }
        ],
    }
    assert decline_when_all_document_packets_irrelevant(state) is None


def test_relevance_decline_when_strong_scores_but_confidently_irrelevant():
    from langgraph_runtime.workflows.document_scope import decline_when_all_document_packets_irrelevant

    packet_id = "retrieval_worker:1"
    state = {
        "corrective_policy": {"minimum_relevance_confidence": 0.65},
        "evidence_packets": [
            {
                "id": packet_id,
                "source_ids": [f"doc:{'a' * 32}:chunk-1"],
                "content": "The Transformer has six encoder layers.",
                "raw_retriever_score": 0.88,
            }
        ],
        "evidence_assessments": [
            {
                "packet_id": packet_id,
                "relevant": False,
                "confidence": 0.95,
            }
        ],
        "question": "What is the population of Tokyo?",
    }
    assert decline_when_all_document_packets_irrelevant(state) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_postprocess_normalizes_attachment_scope_hedges_to_canonical_decline():
    state = {
        "workflow_id": "router_rag_agent",
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "a" * 32}],
            "document_evidence_text": "Attention Is All You Need presents the Transformer.",
        },
        "evidence_packets": [
            {
                "content": "Attention Is All You Need presents the Transformer.",
                "source_ids": [f"doc:{'a' * 32}:chunk-1"],
            }
        ],
        "question": "What is the population of Tokyo?",
    }
    hedge = (
        "I am sorry, but the provided context does not contain information about "
        "the population of Tokyo. Therefore, I cannot answer your question."
    )
    assert postprocess_grounded_document_answer(state, hedge) == DOCUMENT_SCOPE_DECLINE_ANSWER