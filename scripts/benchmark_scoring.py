"""Shared RAG eval scoring (used by promptfoo provider and rag_eval_suite)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

import httpx

DECLINE_RE = re.compile(
    r"attached documents do not|do not address|not (?:found|present|available) in the",
    re.I,
)
INTERNAL_KNOWLEDGE_RE = re.compile(r"\(internal knowledge", re.I)
CITATION_HINT_RE = re.compile(r"\[source:|src_", re.I)

JUDGE_SYSTEM_PROMPT = """You grade RAG answers against reference constraints.
Reply with JSON only: {"pass": true|false, "reason": "..."}.
Pass when the answer satisfies the constraints described in the user message."""

CATEGORY_LABELS = {
    "overview": "overview",
    "fact": "facts",
    "table": "tables",
    "method": "method",
    "multi_doc": "multi_doc",
    "out_of_scope": "out_of_scope",
    "results": "results",
    "conversation": "conversation",
    "regression": "regression",
}


def numeric_score(answer: str, spec: dict[str, Any] | None) -> bool | None:
    if not spec:
        return None
    target = float(spec["target"])
    tol = float(spec.get("tolerance") or 0)
    found = [float(x) for x in re.findall(r"-?\d+\.?\d*", answer)]
    if not found:
        return False
    return any(abs(v - target) <= tol for v in found)


def string_constraints(answer: str, match: dict[str, Any] | None) -> bool | None:
    if not isinstance(match, dict) or not match:
        return None
    lowered = answer.lower()
    if match.get("decline"):
        return bool(DECLINE_RE.search(answer))
    if match.get("none_contains"):
        return not any(sub.lower() in lowered for sub in match["none_contains"])
    if match.get("all_contains"):
        return all(sub.lower() in lowered for sub in match["all_contains"])
    if match.get("any_contains"):
        return any(sub.lower() in lowered for sub in match["any_contains"])
    return None


def grounding_pass(answer: str, *, require_citation: bool = True) -> bool:
    if INTERNAL_KNOWLEDGE_RE.search(answer):
        return False
    if require_citation and not CITATION_HINT_RE.search(answer):
        if DECLINE_RE.search(answer):
            return True
        return len(answer.strip()) < 40
    return True


def llm_judge(
    client: httpx.Client,
    question: str,
    answer: str,
    match: dict[str, Any],
    *,
    judge_model: str,
    completions_url: str = "http://127.0.0.1:8000/v1/chat/completions",
) -> bool:
    user = json.dumps({"question": question, "answer": answer, "constraints": match})
    payload = {
        "model": judge_model,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    r = client.post(completions_url, json=payload, timeout=180.0)
    if r.status_code >= 400:
        return string_constraints(answer, match) is True
    text = r.json()["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(text[text.find("{") : text.rfind("}") + 1])
        return bool(parsed.get("pass"))
    except (json.JSONDecodeError, ValueError):
        return False


def score_answer(
    question: str,
    answer: str,
    case: dict[str, Any],
    *,
    judge: Callable[[str, str, dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    match = case.get("answer_match") or {}
    numeric = case.get("numeric")
    constrained = string_constraints(answer, match)
    numeric_ok = numeric_score(answer, numeric if isinstance(numeric, dict) else None)
    if constrained is not None:
        answer_pass = constrained
    elif numeric_ok is not None:
        answer_pass = numeric_ok
    elif judge is not None:
        answer_pass = judge(question, answer, match)
    else:
        answer_pass = False
    assertions = case.get("assertions") if isinstance(case.get("assertions"), dict) else {}
    if assertions.get("no_internal_knowledge_note"):
        answer_pass = answer_pass and not INTERNAL_KNOWLEDGE_RE.search(answer)
    grounding = grounding_pass(answer, require_citation=bool(case.get("require_citation", True)))
    return {
        "answer_pass": answer_pass,
        "grounding_pass": grounding,
        "answer_heuristic": constrained if constrained is not None else numeric_ok,
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 4)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    latencies = sorted(float(r.get("latency_s") or 0) for r in rows)
    by_category: dict[str, Any] = {}
    for category in sorted({r.get("category") for r in rows if r.get("category")}):
        subset = [r for r in rows if r.get("category") == category]
        by_category[category] = {
            "cases": len(subset),
            "retrieval_hit_rate": _rate(subset, "retrieval_hit"),
            "answer_pass_rate": _rate(subset, "answer_pass"),
            "grounding_pass_rate": _rate(subset, "grounding_pass"),
            "decline_correct_rate": _rate(subset, "decline_correct"),
            "median_latency_s": sorted(float(r.get("latency_s") or 0) for r in subset)[len(subset) // 2]
            if subset
            else 0,
            "total_llm_tokens": sum(int(r.get("llm_tokens_total") or 0) for r in subset),
        }
    return {
        "cases": len(rows),
        "retrieval_hit_rate": round(sum(1 for r in rows if r.get("retrieval_hit")) / n, 4),
        "answer_pass_rate": round(sum(1 for r in rows if r.get("answer_pass")) / n, 4),
        "grounding_pass_rate": round(sum(1 for r in rows if r.get("grounding_pass")) / n, 4),
        "decline_correct_rate": _rate(rows, "decline_correct"),
        "median_latency_s": latencies[len(latencies) // 2] if latencies else 0,
        "total_llm_tokens": sum(int(r.get("llm_tokens_total") or 0) for r in rows),
        "by_category": by_category,
    }


def default_judge_model() -> str:
    return os.getenv(
        "ASKPDF_BENCHMARK_JUDGE_MODEL",
        os.getenv("PROMPTFOO_JUDGE_MODEL", "inclusionai/ling-3.2-flash:free"),
    ).removeprefix("openrouter:")