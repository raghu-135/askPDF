from __future__ import annotations

import re
from typing import Any, Mapping

from langgraph_runtime.workflows.evidence import evidence_packets
from langgraph_runtime.workflows.parallel_contracts import (
    PARALLEL_REFERENCE_WORKFLOW_ID,
)

ROUTER_RAG_WORKFLOW_ID = "router_rag_agent"
DOCUMENT_SCOPE_DECLINE_ANSWER = "The attached documents do not address this question."
ATTACHMENT_IRRELEVANT_RETRIEVAL_SCORE = 0.42

def thread_has_attached_documents(state: Mapping[str, Any]) -> bool:
    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), dict) else {}
    documents = bundle.get("documents")
    if isinstance(documents, list) and documents:
        return True
    if state.get("document_sources"):
        return True
    return False


def _score_from_mapping(source: Mapping[str, Any]) -> float | None:
    for key in ("rerank_score", "score", "raw_score"):
        value = source.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def document_retrieval_scores(state: Mapping[str, Any]) -> list[float]:
    """Collect best-known document retrieval scores from prefetch and worker evidence."""

    scores: list[float] = []
    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), dict) else {}
    for container in (
        bundle.get("document_sources") or [],
        state.get("document_sources") or [],
        (bundle.get("matches") if isinstance(bundle.get("matches"), list) else []),
    ):
        for item in container:
            if not isinstance(item, Mapping):
                continue
            score = _score_from_mapping(item)
            if score is not None:
                scores.append(score)
    for packet in state.get("evidence_packets") or []:
        if not isinstance(packet, Mapping):
            continue
        if not any(str(source_id).startswith("doc:") for source_id in (packet.get("source_ids") or [])):
            continue
        raw = packet.get("raw_retriever_score")
        if isinstance(raw, (int, float)):
            scores.append(float(raw))
            continue
        refs = packet.get("refs") if isinstance(packet.get("refs"), Mapping) else {}
        source = refs.get("source") if isinstance(refs.get("source"), Mapping) else {}
        score = _score_from_mapping(source)
        if score is not None:
            scores.append(score)
    return scores


def attachment_retrieval_scope_decline(state: Mapping[str, Any]) -> str | None:
    """Decline when attached PDFs were searched but nothing scores as plausibly relevant."""

    if not thread_has_attached_documents(state):
        return None
    scores = document_retrieval_scores(state)
    if scores and max(scores) < ATTACHMENT_IRRELEVANT_RETRIEVAL_SCORE:
        return DOCUMENT_SCOPE_DECLINE_ANSWER
    return None


def _salient_proper_nouns(question: str) -> list[str]:
    return [
        match.group(0).lower()
        for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", str(question or ""))
    ]


def _combined_document_evidence_text(state: Mapping[str, Any]) -> str:
    parts: list[str] = []
    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), dict) else {}
    if str(bundle.get("document_evidence_text") or "").strip():
        parts.append(str(bundle["document_evidence_text"]))
    if str(state.get("evidence") or "").strip():
        parts.append(str(state["evidence"]))
    for packet in evidence_packets(dict(state)):
        content = str(packet.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts).lower()


def lexical_attachment_scope_decline(state: Mapping[str, Any]) -> str | None:
    """Decline when the question names entities absent from retrieved document text."""

    proper_nouns = _salient_proper_nouns(str(state.get("question") or ""))
    if not proper_nouns:
        return None
    haystack = _combined_document_evidence_text(state)
    if not haystack.strip():
        return DOCUMENT_SCOPE_DECLINE_ANSWER
    if any(term not in haystack for term in proper_nouns):
        return DOCUMENT_SCOPE_DECLINE_ANSWER
    return None


def post_retrieval_attachment_decline(state: Mapping[str, Any]) -> str | None:
    """Score- and evidence-based decline after workers have run."""

    if not thread_has_attached_documents(state):
        return None
    decline = attachment_retrieval_scope_decline(state)
    if decline:
        return decline
    return lexical_attachment_scope_decline(state)


def workflow_post_retrieval_scope_decline(state: Mapping[str, Any]) -> str | None:
    workflow_id = str(state.get("workflow_id") or "")
    if workflow_id not in {ROUTER_RAG_WORKFLOW_ID, PARALLEL_REFERENCE_WORKFLOW_ID}:
        return None
    if workflow_id == PARALLEL_REFERENCE_WORKFLOW_ID:
        from langgraph_runtime.workflows.corrective_nodes import _non_document_evidence_packets

        if _non_document_evidence_packets(state):
            return None
    return post_retrieval_attachment_decline(state)
