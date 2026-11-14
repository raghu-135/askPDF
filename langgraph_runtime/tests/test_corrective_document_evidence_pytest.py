from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from langgraph_runtime.workflows.answer_nodes import finalizer_node
from langgraph_runtime.workflows.corrective_contracts import CORRECTIVE_WORKFLOW_ID
from langgraph_runtime.workflows.corrective_nodes import (
    DOCUMENT_SCOPE_DECLINE_ANSWER,
    corrective_document_scope_decline,
    grounded_route_for_report,
    synthesized_answer_grounded_in_corrective_evidence,
)
from langgraph_runtime.workflows.evidence import (
    append_corrective_evidence_packets,
    corrective_evidence_packets,
)
from langgraph_runtime.workflows.workers import run_tool_worker, tool_worker_spec
from langgraph_runtime.workflows.enums import WorkflowNodeType

def test_corrective_retrieval_worker_packets_from_document_segments():
    source_id = "doc:abcdefabcdefabcdefabcdefabcdefab:c-table2"
    segments = [
        {
            "source_id": source_id,
            "kind": "document",
            "content": "Base model training cost is 3.3 x 10^18 FLOPs.",
            "display": {"title": "attention.pdf"},
            "locator": {
                "file_hash": "abcdefabcdefabcdefabcdefabcdefab",
                "chunk_id": "c-table2",
            },
            "raw_score": 0.92,
        }
    ]
    state = {
        "workflow_id": CORRECTIVE_WORKFLOW_ID,
        "evidence_packets": [],
        "corrective_policy": {"minimum_relevance_confidence": 0.65},
    }
    config = {
        "configurable": {
            "langgraph_node": "retrieval_worker",
            "langgraph_node_type": WorkflowNodeType.RETRIEVAL_WORKER.value,
            "langgraph_visit_index": 1,
        }
    }
    packets = append_corrective_evidence_packets(state, config, segments=segments)
    assert len(packets) == 1
    assert packets[0]["source_ids"] == [source_id]
    graded = {
        **state,
        "evidence_packets": packets,
        "evidence_assessments": [
            {
                "packet_id": packets[0]["id"],
                "relevant": True,
                "confidence": 0.9,
                "provenance_complete": True,
                "instruction_injection_risk": False,
                "coverage": ["training cost"],
            }
        ],
    }
    eligible = corrective_evidence_packets(graded)
    assert len(eligible) == 1
    assert "3.3" in eligible[0]["content"]


@pytest.mark.asyncio
async def test_corrective_retrieval_worker_run_builds_packets_from_tool_artifacts():
    source_id = "doc:abcdefabcdefabcdefabcdefabcdefab:c-table2"
    payload = {
        "ok": True,
        "content": "Table 2 cost row",
        "artifacts": {
            "document_sources": [{"file_hash": "abcdefabcdefabcdefabcdefabcdefab", "chunk_id": "c-table2"}],
            "evidence_segments": [
                {
                    "source_id": source_id,
                    "kind": "document",
                    "content": "Base model training cost is 3.3 x 10^18 FLOPs.",
                    "display": {"title": "attention.pdf"},
                    "locator": {"file_hash": "abcdefabcdefabcdefabcdefabcdefab", "chunk_id": "c-table2"},
                }
            ],
        },
    }
    state = {
        "workflow_id": CORRECTIVE_WORKFLOW_ID,
        "question": "What is the base model training cost in Table 2?",
        "evidence_packets": [],
        "document_sources": [],
        "corrective_policy": {},
    }
    config = {
        "configurable": {
            "langgraph_node": "retrieval_worker",
            "langgraph_node_type": WorkflowNodeType.RETRIEVAL_WORKER.value,
            "langgraph_visit_index": 1,
        }
    }
    from langgraph_runtime.workflows.evidence import combine_evidence, evidence_text_limit
    from langgraph_runtime.workflows.trace import compact_preview, compact_refs, normalize_warnings

    spec = tool_worker_spec(WorkflowNodeType.RETRIEVAL_WORKER.value)
    result = await run_tool_worker(
        state,
        config,
        started=0.0,
        spec=spec,
        should_skip_worker=lambda *_args: False,
        skipped_worker_update=lambda *_args: {},
        tool_config_for_node=lambda *_args, **_kwargs: config,
        invoke_tool_for_node=AsyncMock(return_value=payload),
        normalize_tool_result=lambda raw, **_kwargs: raw,
        combine_evidence=combine_evidence,
        evidence_text_limit=evidence_text_limit,
        append_evidence_packet=lambda *_args, **_kwargs: [],
        refs_from_artifacts=lambda artifacts: {"document_sources": artifacts.get("document_sources", [])},
        state_evidence_refs=lambda *_args: {},
        compact_refs=compact_refs,
        compact_preview=compact_preview,
        normalize_warnings=normalize_warnings,
        log_node_end=lambda *_args: None,
        append_event=lambda *_args, **_kwargs: [],
        append_tool_event=lambda *_args, **_kwargs: [],
    )
    assert len(result["evidence_packets"]) == 1
    assert result["evidence_packets"][0]["source_ids"] == [source_id]
    assert result["document_sources"]


def test_corrective_document_scope_decline_when_document_hits_are_irrelevant():
    packet_id = "retrieval_worker:visit:1:document:abc"
    source_id = "doc:abcdefabcdefabcdefabcdefabcdefab:c-intro"
    state = {
        "workflow_id": CORRECTIVE_WORKFLOW_ID,
        "corrective_policy": {"minimum_relevance_confidence": 0.65},
        "evidence_packets": [
            {
                "id": packet_id,
                "source_ids": [source_id],
                "content": "We propose the Transformer architecture.",
            }
        ],
        "evidence_assessments": [
            {
                "packet_id": packet_id,
                "relevant": False,
                "confidence": 0.2,
                "provenance_complete": True,
                "instruction_injection_risk": False,
            }
        ],
    }
    assert corrective_document_scope_decline(state) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_corrective_document_scope_declines_tokyo_when_only_conversation_evidence_is_eligible():
    packet_id = "thread_conversation_history_worker:visit:1:conversation:1"
    state = {
        "workflow_id": CORRECTIVE_WORKFLOW_ID,
        "pre_fetch_bundle": {"documents": [{"file_hash": "abc" * 8}]},
        "question": "What is the population of Tokyo?",
        "evidence_packets": [
            {
                "id": packet_id,
                "source_ids": ["conversation:61f46c14"],
                "content": "Earlier you asked about Tokyo demographics.",
            }
        ],
        "evidence_assessments": [
            {
                "packet_id": packet_id,
                "relevant": True,
                "confidence": 0.9,
                "provenance_complete": True,
                "instruction_injection_risk": False,
            }
        ],
        "corrective_policy": {"minimum_relevance_confidence": 0.65},
    }
    assert corrective_document_scope_decline(state) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_corrective_document_scope_decline_skips_when_relevant_packet_exists():
    packet_id = "retrieval_worker:visit:1:document:abc"
    source_id = "doc:abcdefabcdefabcdefabcdefabcdefab:c-table2"
    state = {
        "workflow_id": CORRECTIVE_WORKFLOW_ID,
        "corrective_policy": {"minimum_relevance_confidence": 0.65},
        "evidence_packets": [
            {
                "id": packet_id,
                "source_ids": [source_id],
                "content": "Base model training cost is 3.3 x 10^18 FLOPs.",
            }
        ],
        "evidence_assessments": [
            {
                "packet_id": packet_id,
                "relevant": True,
                "confidence": 0.95,
                "provenance_complete": True,
                "instruction_injection_risk": False,
                "coverage": ["training cost"],
            }
        ],
    }
    assert corrective_document_scope_decline(state) is None
    assert corrective_evidence_packets(state)


def test_synthesized_title_answer_grounded_in_corrective_evidence():
    packet_id = "retrieval_worker:visit:1:document:abc"
    source_id = "doc:abcdefabcdefabcdefabcdefabcdefab:c-title"
    state = {
        "workflow_id": CORRECTIVE_WORKFLOW_ID,
        "final_answer": "Attention Is All You Need",
        "corrective_policy": {"minimum_relevance_confidence": 0.65},
        "evidence_packets": [
            {
                "id": packet_id,
                "source_ids": [source_id],
                "content": "Provided proper attribution, Attention Is All You Need is provided.",
            }
        ],
        "evidence_assessments": [
            {
                "packet_id": packet_id,
                "relevant": True,
                "confidence": 0.95,
                "provenance_complete": True,
                "instruction_injection_risk": False,
                "coverage": ["title"],
            }
        ],
    }
    assert synthesized_answer_grounded_in_corrective_evidence(state, state["final_answer"])
    route, _ = grounded_route_for_report({"supported_claim_ratio": 0.0, "unresolved_gaps": ["title"]}, state)
    assert route == "pass"


@pytest.mark.asyncio
async def test_finalizer_preserves_grounded_prose_when_verifier_emits_no_claims():
    packet_id = "retrieval_worker:visit:1:document:abc"
    source_id = "doc:abcdefabcdefabcdefabcdefabcdefab:c-title"
    state = {
        "workflow_id": CORRECTIVE_WORKFLOW_ID,
        "grounded_answer_route": "finalize_cautious",
        "corrective_retrieval_route": "synthesize",
        "final_answer": "Attention Is All You Need",
        "verified_claims": [],
        "contradiction_report": [],
        "evidence_packets": [
            {
                "id": packet_id,
                "source_ids": [source_id],
                "content": "Attention Is All You Need",
            }
        ],
        "evidence_assessments": [
            {
                "packet_id": packet_id,
                "relevant": True,
                "confidence": 0.95,
                "provenance_complete": True,
                "instruction_injection_risk": False,
                "coverage": ["title"],
            }
        ],
        "corrective_policy": {"minimum_relevance_confidence": 0.65},
        "unresolved_gaps": ["paper title"],
    }
    result = await finalizer_node(state, {"configurable": {"langgraph_node": "finalizer"}})
    assert result["final_answer"] == "Attention Is All You Need"
    assert "could not verify" not in result["final_answer"].casefold()
    assert "Unresolved evidence gaps" not in result["final_answer"]


@pytest.mark.asyncio
async def test_finalizer_declines_out_of_scope_corrective_questions():
    packet_id = "retrieval_worker:visit:1:document:abc"
    source_id = "doc:abcdefabcdefabcdefabcdefabcdefab:c-intro"
    state = {
        "workflow_id": CORRECTIVE_WORKFLOW_ID,
        "grounded_answer_route": "finalize_cautious",
        "corrective_retrieval_route": "insufficient",
        "verified_claims": [],
        "contradiction_report": [],
        "evidence_packets": [
            {
                "id": packet_id,
                "source_ids": [source_id],
                "content": "We propose the Transformer architecture.",
            }
        ],
        "evidence_assessments": [
            {
                "packet_id": packet_id,
                "relevant": False,
                "confidence": 0.1,
                "provenance_complete": True,
                "instruction_injection_risk": False,
            }
        ],
        "corrective_policy": {"minimum_relevance_confidence": 0.65},
    }
    result = await finalizer_node(state, {"configurable": {"langgraph_node": "finalizer"}})
    assert result["final_answer"] == DOCUMENT_SCOPE_DECLINE_ANSWER
