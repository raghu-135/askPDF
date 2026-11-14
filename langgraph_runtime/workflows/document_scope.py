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

_LEXICAL_PROPER_NOUN_STOPWORDS = frozenset({
    "attention",
    "bleu",
    "decoder",
    "encoder",
    "flops",
    "for",
    "from",
    "how",
    "layer",
    "layers",
    "model",
    "reply",
    "score",
    "table",
    "the",
    "training",
    "transformer",
    "what",
    "when",
    "where",
    "which",
    "why",
    "with",
})

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


def document_evidence_packets(state: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        packet
        for packet in evidence_packets(dict(state))
        if isinstance(packet, Mapping)
        and any(str(source_id).startswith("doc:") for source_id in (packet.get("source_ids") or []))
    ]


def _packet_retrieval_score(packet: Mapping[str, Any]) -> float | None:
    raw = packet.get("raw_retriever_score")
    if isinstance(raw, (int, float)):
        return float(raw)
    refs = packet.get("refs") if isinstance(packet.get("refs"), Mapping) else {}
    source = refs.get("source") if isinstance(refs.get("source"), Mapping) else {}
    return _score_from_mapping(source)


def worker_document_retrieval_scores(state: Mapping[str, Any]) -> list[float]:
    return [
        score
        for packet in document_evidence_packets(state)
        if (score := _packet_retrieval_score(packet)) is not None
    ]


def prefetch_document_retrieval_scores(state: Mapping[str, Any]) -> list[float]:
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
    return scores


def document_retrieval_scores(state: Mapping[str, Any]) -> list[float]:
    """Collect best-known document retrieval scores from prefetch and worker evidence."""

    return [*prefetch_document_retrieval_scores(state), *worker_document_retrieval_scores(state)]


def attachment_retrieval_scope_decline(state: Mapping[str, Any]) -> str | None:
    """Decline when attached PDFs were searched but nothing scores as plausibly relevant."""

    if not thread_has_attached_documents(state):
        return None
    doc_packets = document_evidence_packets(state)
    worker_scores = worker_document_retrieval_scores(state)
    if doc_packets:
        if worker_scores:
            if max(worker_scores) >= ATTACHMENT_IRRELEVANT_RETRIEVAL_SCORE:
                return None
            if any(str(packet.get("content") or "").strip() for packet in doc_packets):
                return None
            return DOCUMENT_SCOPE_DECLINE_ANSWER
        if any(str(packet.get("content") or "").strip() for packet in doc_packets):
            return None
    prefetch_scores = prefetch_document_retrieval_scores(state)
    if prefetch_scores and max(prefetch_scores) < ATTACHMENT_IRRELEVANT_RETRIEVAL_SCORE:
        return DOCUMENT_SCOPE_DECLINE_ANSWER
    return None


def _salient_proper_nouns(question: str) -> list[str]:
    return [
        term
        for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", str(question or ""))
        if (term := match.group(0).lower()) not in _LEXICAL_PROPER_NOUN_STOPWORDS
    ]


def _document_attachment_evidence_text(state: Mapping[str, Any]) -> str:
    """Document-sourced text only, excluding conversation history and synthetic context."""

    parts: list[str] = []
    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), dict) else {}
    if str(bundle.get("document_evidence_text") or "").strip():
        parts.append(str(bundle["document_evidence_text"]))
    for packet in document_evidence_packets(state):
        content = str(packet.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts).lower()


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
    haystack = _document_attachment_evidence_text(state)
    if not haystack.strip():
        return DOCUMENT_SCOPE_DECLINE_ANSWER
    if any(term not in haystack for term in proper_nouns):
        return DOCUMENT_SCOPE_DECLINE_ANSWER
    return None


def _retrieved_document_evidence_is_substantive(state: Mapping[str, Any]) -> bool:
    doc_packets = document_evidence_packets(state)
    worker_scores = worker_document_retrieval_scores(state)
    if worker_scores and max(worker_scores) >= ATTACHMENT_IRRELEVANT_RETRIEVAL_SCORE:
        return True
    if any(str(packet.get("content") or "").strip() for packet in doc_packets):
        return True
    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), dict) else {}
    if str(bundle.get("document_evidence_text") or "").strip():
        return True
    if str(state.get("evidence") or "").strip():
        return True
    prefetch_scores = prefetch_document_retrieval_scores(state)
    return bool(doc_packets and prefetch_scores)


def post_retrieval_attachment_decline(state: Mapping[str, Any]) -> str | None:
    """Score- and evidence-based decline after workers have run."""

    if not thread_has_attached_documents(state):
        return None
    if _retrieved_document_evidence_is_substantive(state):
        return lexical_attachment_scope_decline(state)
    decline = attachment_retrieval_scope_decline(state)
    if decline:
        return decline
    return lexical_attachment_scope_decline(state)


def workflow_post_retrieval_scope_decline(state: Mapping[str, Any]) -> str | None:
    workflow_id = str(state.get("workflow_id") or "")
    if workflow_id not in {ROUTER_RAG_WORKFLOW_ID, PARALLEL_REFERENCE_WORKFLOW_ID}:
        return None
    return post_retrieval_attachment_decline(state)
