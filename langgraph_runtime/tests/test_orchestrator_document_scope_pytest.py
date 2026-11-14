from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from langgraph_runtime.workflows.answer_nodes import answer_from_context_node
from langgraph_runtime.workflows.corrective_nodes import (
    DOCUMENT_SCOPE_DECLINE_ANSWER,
    document_scope_decline,
    orchestrator_document_scope_decline,
)
from langgraph_runtime.workflows.enums import WorkflowNodeType
from langgraph_runtime.workflows.parallel_contracts import (
    PARALLEL_REFERENCE_WORKFLOW_ID,
    PARALLEL_WORKER_PROGRESS_EXTENSION_SECONDS,
    PARALLEL_WORKER_PROGRESS_MAX_EXTENSIONS,
    parallel_timeout_watchdog_seconds,
)
from langgraph_runtime.compiler import WorkflowCompiler


def test_parallel_timeout_watchdog_includes_progress_extension_budget():
    worker_ms = 90_000
    attempts = 2
    expected = (
        (worker_ms / 1000 + PARALLEL_WORKER_PROGRESS_MAX_EXTENSIONS * PARALLEL_WORKER_PROGRESS_EXTENSION_SECONDS)
        * attempts
        + 0.25
        + 1.0
    )
    assert parallel_timeout_watchdog_seconds(worker_ms, attempts) == pytest.approx(expected, rel=1e-6)


def test_orchestrator_declines_when_document_chunks_score_below_threshold():
    state = {
        "workflow_id": PARALLEL_REFERENCE_WORKFLOW_ID,
        "pre_fetch_bundle": {"documents": [{"file_hash": "abc" * 8}]},
        "evidence_packets": [
            {
                "id": "retrieval_worker:1",
                "source_ids": ["doc:" + "ab" * 16 + ":c1"],
                "content": "",
                "raw_retriever_score": 0.21,
            }
        ],
    }
    assert orchestrator_document_scope_decline(state) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_orchestrator_tokyo_declines_when_assessment_marks_unrelated_packet_relevant():
    state = {
        "workflow_id": PARALLEL_REFERENCE_WORKFLOW_ID,
        "pre_fetch_bundle": {"documents": [{"file_hash": "abc" * 8}]},
        "question": "What is the population of Tokyo?",
        "evidence_packets": [
            {
                "id": "retrieval_worker:1",
                "source_ids": ["doc:" + "ab" * 16 + ":c1"],
                "content": "The Transformer has six encoder layers.",
                "raw_retriever_score": 0.62,
            }
        ],
        "evidence_assessments": [
            {"packet_id": "retrieval_worker:1", "relevant": True},
        ],
    }
    assert orchestrator_document_scope_decline(state) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_orchestrator_keeps_in_scope_when_assessment_marks_doc_packet_relevant():
    state = {
        "workflow_id": PARALLEL_REFERENCE_WORKFLOW_ID,
        "pre_fetch_bundle": {"documents": [{"file_hash": "abc" * 8}]},
        "evidence_packets": [
            {
                "id": "retrieval_worker:1",
                "source_ids": ["doc:" + "ab" * 16 + ":c1"],
                "content": "Table 2 big-model EN-FR training cost is 2.3 x 10^19 FLOPs.",
                "raw_retriever_score": 0.21,
            }
        ],
        "evidence_assessments": [
            {"packet_id": "retrieval_worker:1", "relevant": True},
        ],
    }
    assert orchestrator_document_scope_decline(state) is None


def test_orchestrator_tokyo_canonical_decline_when_context_loader_has_document_hits():
    state = {
        "workflow_id": PARALLEL_REFERENCE_WORKFLOW_ID,
        "pre_fetch_bundle": {
            "documents": [{"file_hash": "abc" * 8}],
            "document_evidence_text": "Attention paper discusses transformer encoder layers.",
            "document_sources": [{"file_hash": "abc" * 8, "score": 0.55}],
        },
        "document_sources": [{"file_hash": "abc" * 8, "score": 0.55}],
        "question": "What is the population of Tokyo?",
        "evidence_packets": [],
    }
    assert orchestrator_document_scope_decline(state) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_orchestrator_tokyo_declines_canonical_despite_non_document_packets():
    state = {
        "workflow_id": PARALLEL_REFERENCE_WORKFLOW_ID,
        "pre_fetch_bundle": {"documents": [{"file_hash": "abc" * 8}]},
        "question": "What is the population of Tokyo?",
        "evidence_packets": [
            {
                "id": "retrieval_worker:1",
                "source_ids": ["doc:" + "ab" * 16 + ":c1"],
                "content": "The Transformer has six encoder layers.",
                "raw_retriever_score": 0.55,
            },
            {
                "id": "durable_memory_worker:1",
                "source_ids": ["memory:mem-1"],
                "content": "User prefers concise answers.",
            },
        ],
    }
    assert orchestrator_document_scope_decline(state) == DOCUMENT_SCOPE_DECLINE_ANSWER


def test_orchestrator_keeps_encoder_in_scope_when_only_combined_evidence_text():
    state = {
        "workflow_id": PARALLEL_REFERENCE_WORKFLOW_ID,
        "pre_fetch_bundle": {"documents": [{"file_hash": "abc" * 8}]},
        "question": "How many encoder layers does the base Transformer use?",
        "evidence": "Retrieved: The base model uses N=6 identical layers.",
        "evidence_packets": [
            {
                "id": "retrieval_worker:1",
                "source_ids": ["doc:" + "ab" * 16 + ":c1"],
                "content": "",
            }
        ],
        "evidence_assessments": [
            {"packet_id": "retrieval_worker:1", "relevant": False},
        ],
    }
    assert orchestrator_document_scope_decline(state) is None


def test_orchestrator_keeps_encoder_in_scope_when_assessments_mark_irrelevant():
    state = {
        "workflow_id": PARALLEL_REFERENCE_WORKFLOW_ID,
        "pre_fetch_bundle": {"documents": [{"file_hash": "abc" * 8}]},
        "question": "How many encoder layers does the base Transformer use?",
        "evidence_packets": [
            {
                "id": "retrieval_worker:1",
                "source_ids": ["doc:" + "ab" * 16 + ":c1"],
                "content": "The base model uses N=6 identical layers for encoder and decoder.",
                "raw_retriever_score": 0.35,
            }
        ],
        "evidence_assessments": [
            {"packet_id": "retrieval_worker:1", "relevant": False},
        ],
    }
    assert orchestrator_document_scope_decline(state) is None


def test_orchestrator_keeps_in_scope_when_a_chunk_scores_above_threshold():
    state = {
        "workflow_id": PARALLEL_REFERENCE_WORKFLOW_ID,
        "pre_fetch_bundle": {"documents": [{"file_hash": "abc" * 8}]},
        "evidence_packets": [
            {
                "id": "retrieval_worker:1",
                "source_ids": ["doc:" + "ab" * 16 + ":c1"],
                "content": "Table 2 training cost for the base model is 3.3 x 10^18 FLOPs.",
                "raw_retriever_score": 0.88,
            }
        ],
    }
    assert orchestrator_document_scope_decline(state) is None


def test_document_scope_decline_routes_by_workflow_id():
    corrective_state = {
        "workflow_id": "corrective_self_rag_agent",
        "evidence_packets": [],
    }
    assert document_scope_decline(corrective_state) is None

    orchestrator_state = {
        "workflow_id": PARALLEL_REFERENCE_WORKFLOW_ID,
        "pre_fetch_bundle": {"documents": [{"file_hash": "x" * 32}]},
        "evidence_packets": [],
        "errors": [{"code": "document_scope_forbidden", "message": "forbidden"}],
    }
    assert document_scope_decline(orchestrator_state) == DOCUMENT_SCOPE_DECLINE_ANSWER


@pytest.mark.asyncio
async def test_synthesizer_tokyo_canonical_decline_with_non_document_packets():
    state = {
        "workflow_id": PARALLEL_REFERENCE_WORKFLOW_ID,
        "pre_fetch_bundle": {"documents": [{"file_hash": "abc" * 8}]},
        "question": "What is the population of Tokyo?",
        "evidence_packets": [
            {
                "id": "retrieval_worker:1",
                "source_ids": ["doc:" + "ab" * 16 + ":c1"],
                "content": "Encoder layers count is six.",
                "raw_retriever_score": 0.62,
            },
            {
                "id": "durable_memory_worker:1",
                "source_ids": ["memory:mem-1"],
                "content": "Prior thread context.",
            },
        ],
        "llm_model": "openai/gpt-4.1-nano",
    }
    config = {"configurable": {"langgraph_node": WorkflowNodeType.SYNTHESIZER.value}}
    with patch("langgraph_runtime.workflows.answer_nodes._get_llm", new=AsyncMock()) as mocked_llm:
        result = await answer_from_context_node(state, config, node_name=WorkflowNodeType.SYNTHESIZER.value)
        mocked_llm.assert_not_called()
    assert result["final_answer"] == DOCUMENT_SCOPE_DECLINE_ANSWER
    assert result.get("document_scope_declined") is True


@pytest.mark.asyncio
async def test_synthesizer_short_circuits_document_scope_decline_without_llm():
    state = {
        "workflow_id": PARALLEL_REFERENCE_WORKFLOW_ID,
        "pre_fetch_bundle": {"documents": [{"file_hash": "abc" * 8}]},
        "evidence_packets": [
            {
                "id": "retrieval_worker:1",
                "source_ids": ["doc:" + "ab" * 16 + ":c1"],
                "content": "Unrelated PDF excerpt.",
                "raw_retriever_score": 0.15,
            }
        ],
        "llm_model": "openai/gpt-4.1-nano",
    }
    config = {"configurable": {"langgraph_node": WorkflowNodeType.SYNTHESIZER.value}}
    with patch("langgraph_runtime.workflows.answer_nodes._get_llm", new=AsyncMock()) as mocked_llm:
        result = await answer_from_context_node(state, config, node_name=WorkflowNodeType.SYNTHESIZER.value)
        mocked_llm.assert_not_called()
    assert result["final_answer"] == DOCUMENT_SCOPE_DECLINE_ANSWER
    assert result.get("document_scope_declined") is True


def test_orchestrator_workflow_compiles_after_scope_changes():
    import json
    from pathlib import Path

    rel = Path("rag_service/app/workflow_catalog/builtins/orchestrator_worker_rag_agent.json")
    spec_path = next(
        (
            candidate
            for candidate in (
                rel,
                Path(__file__).resolve().parents[2] / rel,
            )
            if candidate.is_file()
        ),
        None,
    )
    if spec_path is None:
        pytest.skip("orchestrator workflow catalog is not bundled in the langgraph-runtime image")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    assert WorkflowCompiler().compile(spec) is not None
