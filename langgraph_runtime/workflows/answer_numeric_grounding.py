from __future__ import annotations

import re
from typing import Any, Mapping

from langgraph_runtime.workflows.evidence import evidence_packets, final_context_from_state

_NUMERIC_LITERAL_RE = re.compile(
    r"(?:"
    r"\d[\d,]*(?:\.\d+)?\s*[x×]\s*10\s*(?:\^\s*[-+]?\d+|\*\*[-+]?\d+)"
    r"|\d[\d,]*(?:\.\d+)?[eE][+-]?\d+"
    r"|\d[\d,]*(?:\.\d+)?"
    r")"
)


def _normalize_numeric_literal(value: str) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("×", "x")
    text = re.sub(r"\s+", " ", text)
    return text


def extract_numeric_literals(text: str) -> list[str]:
    normalized_to_raw: dict[str, str] = {}
    for match in _NUMERIC_LITERAL_RE.finditer(str(text or "")):
        raw = match.group(0).strip()
        if not raw:
            continue
        key = _normalize_numeric_literal(raw)
        if key and key not in normalized_to_raw:
            normalized_to_raw[key] = raw
    return list(normalized_to_raw.values())


def _evidence_corpus(state: Mapping[str, Any]) -> str:
    context, _source = final_context_from_state(dict(state))
    parts = [str(context or "")]
    for packet in evidence_packets(dict(state)):
        parts.append(str(packet.get("content") or ""))
    return "\n".join(part for part in parts if part.strip())


def numeric_grounding_violations(answer: str, evidence_text: str) -> list[str]:
    """Return answer numeric literals that do not appear verbatim in the evidence."""

    answer_literals = extract_numeric_literals(answer)
    if not answer_literals:
        return []
    evidence_keys = {_normalize_numeric_literal(item) for item in extract_numeric_literals(evidence_text)}
    violations: list[str] = []
    for literal in answer_literals:
        if _normalize_numeric_literal(literal) not in evidence_keys:
            violations.append(literal)
    return violations


def answer_numeric_grounding_issues(state: Mapping[str, Any]) -> list[str]:
    answer = str(state.get("final_answer") or "")
    if not answer.strip():
        return []
    violations = numeric_grounding_violations(answer, _evidence_corpus(state))
    if not violations:
        return []
    joined = ", ".join(violations)
    return [
        "Answer numeric literals must match evidence verbatim (no magnitude reformatting): "
        f"{joined}"
    ]
