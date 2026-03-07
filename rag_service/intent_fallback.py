import re
from typing import Any, Dict, List, Optional


_FOLLOWUP_PREFIXES = (
    "what about",
    "and",
    "also",
    "that",
    "it",
    "this",
)

_PRONOUN_PAT = re.compile(r"\b(it|this|that|they|them|the document|the paper|the pdf|the webpage)\b", re.I)


def _extract_last_user_line(recent_history_text: str) -> Optional[str]:
    if not recent_history_text:
        return None
    lines = [ln.strip() for ln in recent_history_text.splitlines() if ln.strip()]
    for ln in reversed(lines):
        if ln.lower().startswith("user:"):
            return ln.split(":", 1)[1].strip()
    return None


def _is_followup_question(question: str) -> bool:
    q = question.strip().lower()
    if len(q.split()) <= 6:
        for pref in _FOLLOWUP_PREFIXES:
            if q.startswith(pref):
                return True
    return False


def heuristic_rewrite_query(
    question: str,
    prefetch_bundle: Optional[Dict[str, Any]] = None,
) -> str:
    q = question.strip()
    bundle = prefetch_bundle or {}
    documents: List[Dict[str, str]] = bundle.get("documents") or []
    recent_text = bundle.get("recent_history_text") or ""

    if documents and len(documents) == 1:
        doc_name = documents[0].get("file_name") or "the uploaded document"
        if re.search(r"\bsummarize\b", q, re.I):
            return f"Summarize the uploaded document {doc_name}"
        if _PRONOUN_PAT.search(q):
            return f"{q} (in {doc_name})"

    if _is_followup_question(q):
        last_user = _extract_last_user_line(recent_text)
        if last_user:
            return f"{q} (regarding: {last_user})"

    return q
