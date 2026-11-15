from __future__ import annotations

from typing import Any, Dict, Mapping

from langgraph_runtime.workflows.evidence import evidence_packets
from langgraph_runtime.workflows.parallel_contracts import (
    PARALLEL_REFERENCE_WORKFLOW_ID,
)

ROUTER_RAG_WORKFLOW_ID = "router_rag_agent"
DOCUMENT_SCOPE_DECLINE_ANSWER = "The attached documents do not address this question."
ATTACHMENT_IRRELEVANT_RETRIEVAL_SCORE = 0.42
ATTACHMENT_WEAK_RETRIEVAL_SCORE = 0.25


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


def merged_state_document_sources(state: Mapping[str, Any]) -> list[Dict[str, Any]]:
    """Document source rows from prefetch bundle and live worker state."""

    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), dict) else {}
    sources: list[Dict[str, Any]] = []
    for container in (state.get("document_sources") or [], bundle.get("document_sources") or []):
        if not isinstance(container, list):
            continue
        sources.extend(item for item in container if isinstance(item, dict))
    return sources


def packet_assessments_by_id(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(item.get("packet_id")): item
        for item in (state.get("evidence_assessments") or [])
        if isinstance(item, Mapping) and item.get("packet_id")
    }


def _grader_assessment_failed(assessment: Mapping[str, Any]) -> bool:
    needle = "grader did not return"
    for key in ("contradiction_signals", "rejection_reasons"):
        for item in assessment.get(key) or []:
            if needle in str(item).casefold():
                return True
    return False


def _packet_assessment_confidently_irrelevant(
    assessment: Mapping[str, Any],
    *,
    minimum_relevance_confidence: float,
) -> bool:
    if assessment.get("relevant") is not False:
        return False
    if _grader_assessment_failed(assessment):
        return False
    confidence = assessment.get("confidence")
    if confidence is None:
        return True
    try:
        return float(confidence) >= minimum_relevance_confidence
    except (TypeError, ValueError):
        return False


def _answer_is_attachment_scope_hedge(answer: str) -> bool:
    """Detect model hedges that mean attached documents cannot answer the question."""

    text = str(answer or "").strip()
    if not text or text == DOCUMENT_SCOPE_DECLINE_ANSWER:
        return False
    lowered = text.casefold()
    phrases = (
        "does not contain",
        "do not contain",
        "not contain information",
        "cannot answer your question",
        "cannot answer the question",
        "cannot answer this question",
        "can't answer",
        "do not have information",
        "does not have information",
        "don't have information",
        "not in the provided context",
        "not in the attached",
        "provided context does not",
        "i am sorry, but the provided context",
    )
    return any(phrase in lowered for phrase in phrases)


def decline_when_all_document_packets_irrelevant(state: Mapping[str, Any]) -> str | None:
    """Decline when every assessed document packet is confidently marked not relevant."""

    from langgraph_runtime.workflows.corrective_contracts import normalized_corrective_policy

    doc_packets = document_evidence_packets(state)
    assessments = packet_assessments_by_id(state)
    if not doc_packets or not assessments:
        return None
    policy = normalized_corrective_policy(state.get("corrective_policy"))
    minimum_confidence = float(policy["minimum_relevance_confidence"])
    for packet in doc_packets:
        assessment = assessments.get(str(packet.get("id") or ""))
        if not assessment or assessment.get("relevant") is True:
            return None
        if not _packet_assessment_confidently_irrelevant(
            assessment,
            minimum_relevance_confidence=minimum_confidence,
        ):
            return None
    return DOCUMENT_SCOPE_DECLINE_ANSWER


def attachment_retrieval_scope_decline(state: Mapping[str, Any]) -> str | None:
    """Decline when attached PDFs were searched but nothing scores as plausibly relevant."""

    if not thread_has_attached_documents(state):
        return None
    doc_packets = document_evidence_packets(state)
    worker_scores = worker_document_retrieval_scores(state)
    if doc_packets:
        if worker_scores:
            peak_worker = max(worker_scores)
            if peak_worker >= ATTACHMENT_IRRELEVANT_RETRIEVAL_SCORE:
                return None
            if peak_worker >= ATTACHMENT_WEAK_RETRIEVAL_SCORE:
                return None
            if any(str(packet.get("content") or "").strip() for packet in doc_packets):
                return None
            return DOCUMENT_SCOPE_DECLINE_ANSWER
        if any(str(packet.get("content") or "").strip() for packet in doc_packets):
            return None
    prefetch_scores = prefetch_document_retrieval_scores(state)
    if prefetch_scores:
        peak_prefetch = max(prefetch_scores)
        if peak_prefetch >= ATTACHMENT_IRRELEVANT_RETRIEVAL_SCORE:
            return None
        if peak_prefetch < ATTACHMENT_WEAK_RETRIEVAL_SCORE:
            return DOCUMENT_SCOPE_DECLINE_ANSWER
    return None


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


def workflow_post_retrieval_scope_decline(state: Mapping[str, Any]) -> str | None:
    workflow_id = str(state.get("workflow_id") or "")
    if workflow_id not in {ROUTER_RAG_WORKFLOW_ID, PARALLEL_REFERENCE_WORKFLOW_ID}:
        return None
    if not thread_has_attached_documents(state):
        return None
    return attachment_retrieval_scope_decline(state)


def attached_document_evidence_haystack(state: Mapping[str, Any]) -> str:
    """Retrieved document text used for grounding checks across workflows."""

    parts: list[str] = []
    if str(state.get("evidence") or "").strip():
        parts.append(str(state["evidence"]))
    bundle = state.get("pre_fetch_bundle") if isinstance(state.get("pre_fetch_bundle"), dict) else {}
    for container in (bundle.get("matches") or [], bundle.get("document_sources") or []):
        if not isinstance(container, list):
            continue
        for item in container:
            if not isinstance(item, Mapping):
                continue
            for key in ("content", "text", "snippet", "body"):
                chunk = str(item.get(key) or "").strip()
                if chunk:
                    parts.append(chunk)
                    break
    if str(bundle.get("document_evidence_text") or "").strip():
        parts.append(str(bundle["document_evidence_text"]))
    for packet in document_evidence_packets(state):
        content = str(packet.get("content") or "").strip()
        if content:
            parts.append(content)
    for packet in evidence_packets(dict(state)):
        if not isinstance(packet, Mapping):
            continue
        content = str(packet.get("content") or "").strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


def postprocess_grounded_document_answer(state: Mapping[str, Any], answer: str) -> str:
    """Normalize scope declines; never inject question-specific answer facts."""

    text = str(answer or "").strip()
    if not text or not thread_has_attached_documents(state):
        return text
    if text.strip() == DOCUMENT_SCOPE_DECLINE_ANSWER or text.startswith(DOCUMENT_SCOPE_DECLINE_ANSWER + "\n"):
        return DOCUMENT_SCOPE_DECLINE_ANSWER
    relevance_decline = decline_when_all_document_packets_irrelevant(state)
    if relevance_decline:
        return relevance_decline
    if _answer_is_attachment_scope_hedge(text):
        return DOCUMENT_SCOPE_DECLINE_ANSWER
    return text
