"""promptfoo custom provider: one askPDF chat per test (live stack required)."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS))

from benchmark_config import chat_request_options, load_benchmark_config  # noqa: E402
from live_stack_client import (  # noqa: E402
    answer_from_run,
    live_client,
    pages_from_document_sources,
    scored_agent_run,
    thread_chat,
)

CHAT_MODEL = os.getenv("ASKPDF_BENCHMARK_MODEL", "inclusionai/ling-3.2-flash:free")
DECLINE_RE = re.compile(
    r"attached documents do not|do not address|not (?:found|present|available) in the",
    re.I,
)

_JUDGE_MIN_INTERVAL_S = float(os.getenv("ASKPDF_EVAL_MIN_INTERVAL_S", "3.5"))
_LAST_CALL_AT = 0.0


def _throttle() -> None:
    global _LAST_CALL_AT
    now = time.monotonic()
    wait = _JUDGE_MIN_INTERVAL_S - (now - _LAST_CALL_AT)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL_AT = time.monotonic()


def _numeric_score(answer: str, spec: dict[str, Any] | None) -> bool | None:
    if not spec:
        return None
    target = float(spec["target"])
    tol = float(spec.get("tolerance") or 0)
    found = [float(x) for x in re.findall(r"-?\d+\.?\d*", answer)]
    if not found:
        return False
    return any(abs(v - target) <= tol for v in found)


def _string_constraints(answer: str, match: dict[str, Any]) -> bool | None:
    lowered = answer.lower()
    if match.get("decline"):
        return bool(DECLINE_RE.search(answer))
    if match.get("all_contains"):
        return all(sub.lower() in lowered for sub in match["all_contains"])
    if match.get("any_contains"):
        return any(sub.lower() in lowered for sub in match["any_contains"])
    return None


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    vars_ = context.get("vars") or {}
    question = str(vars_.get("question") or prompt).strip()
    thread_id = os.getenv("ASKPDF_EVAL_THREAD_ID", "").strip()
    if not thread_id:
        return {"error": "ASKPDF_EVAL_THREAD_ID not set; run via evals/promptfoo/run_poc.sh"}

    api_base = os.getenv("ASKPDF_BENCHMARK_API", os.getenv("ASKPDF_API_URL", "http://127.0.0.1:8000/api"))
    chat_opts = chat_request_options(load_benchmark_config())

    _throttle()
    try:
        with live_client(api_base=api_base) as client:
            body = thread_chat(
                client,
                thread_id,
                question,
                llm_model=CHAT_MODEL,
                chat_options=chat_opts,
            )
            run_id = body.get("agent_run_id")
            detail = scored_agent_run(client, thread_id, str(run_id)) if run_id else {}
    except httpx.HTTPError as exc:
        return {"error": f"askPDF API error: {exc}"}

    answer = str(body.get("answer") or answer_from_run(detail) or "")
    pages = pages_from_document_sources(body.get("document_sources") or [])
    expected = set(vars_.get("expected_pages") or [])
    retrieval_hit = (not expected) or bool(pages & expected)

    match = vars_.get("answer_match") or {}
    if isinstance(match, str):
        match = json.loads(match)
    numeric = vars_.get("numeric")
    if isinstance(numeric, str) and numeric:
        numeric = json.loads(numeric)
    if numeric == "" or numeric == "null":
        numeric = None

    constrained = _string_constraints(answer, match)
    numeric_ok = _numeric_score(answer, numeric if isinstance(numeric, dict) else None)
    answer_heuristic_pass = (
        constrained if constrained is not None else (numeric_ok if numeric_ok is not None else None)
    )

    return {
        "output": answer,
        "metadata": {
            "question_id": vars_.get("question_id"),
            "retrieval_pages": sorted(pages),
            "expected_pages": sorted(expected),
            "retrieval_hit": retrieval_hit,
            "answer_heuristic_pass": answer_heuristic_pass,
            "workflow": os.getenv("ASKPDF_EVAL_WORKFLOW", "router_rag_agent"),
            "chat_model": CHAT_MODEL,
        },
    }