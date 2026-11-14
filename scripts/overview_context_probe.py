"""Overview-question probe: document_sources vs answer (context vs model)."""

from __future__ import annotations

import json
import re
from typing import Any

TITLE_RE = re.compile(r"attention is all you need", re.I)
ABSTRACT_RE = re.compile(r"\babstract\b", re.I)
MISSING_RE = re.compile(
    r"what is missing|title.*missing|abstract.*missing|not included|does not include|context (?:provided )?mainly",
    re.I,
)


def sources_digest(srcs: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [str(s.get("text") or s.get("preview") or "") for s in srcs]
    joined = "\n".join(texts)
    return {
        "source_count": len(srcs),
        "source_previews": [t[:160] for t in texts],
        "context_has_title_string": bool(TITLE_RE.search(joined)),
        "context_has_abstract_word": bool(ABSTRACT_RE.search(joined)),
    }


def diagnose_answer(answer: str, digest: dict[str, Any]) -> str:
    missing = bool(MISSING_RE.search(answer))
    title_in_answer = bool(TITLE_RE.search(answer))
    if not digest.get("context_has_title_string"):
        if missing:
            return "primarily_context_assembly"
        if title_in_answer:
            return "model_reasoning_despite_thin_context"
        return "context_gap_with_model_hallucination"
    if missing:
        return "primarily_model_reasoning"
    return "healthy"


def probe_row(
    *,
    model: str,
    question: str,
    chat_body: dict[str, Any],
    run_detail: dict[str, Any],
    latency_s: float,
) -> dict[str, Any]:
    answer = str((run_detail.get("final_output") or {}).get("answer") or chat_body.get("answer") or "")
    digest = sources_digest(chat_body.get("document_sources") or [])
    return {
        "model": model,
        "latency_s": latency_s,
        "workflow_id": run_detail.get("workflow_id"),
        "question": question,
        "answer_excerpt": answer[:1500],
        "answer_claims_missing_front_matter": bool(MISSING_RE.search(answer)),
        "answer_mentions_title": bool(TITLE_RE.search(answer)),
        "answer_mentions_abstract": bool(ABSTRACT_RE.search(answer)),
        "document_sources_digest": digest,
        "diagnosis": diagnose_answer(answer, digest),
    }