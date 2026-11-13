from __future__ import annotations

from typing import Any, Mapping

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
