"""promptfoo custom provider (delegates to benchmark_execution)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from benchmark_config import load_benchmark_config, thread_settings_baseline  # noqa: E402
from benchmark_execution import merge_case_chat_options, run_turn  # noqa: E402
from live_stack_client import live_client  # noqa: E402

CHAT_MODEL = os.getenv("ASKPDF_BENCHMARK_MODEL", "nvidia/nemotron-3.5-super-vl:free")
_LAST_CALL_AT = 0.0
_MIN_INTERVAL = float(os.getenv("ASKPDF_EVAL_MIN_INTERVAL_S", "3.5"))


def _throttle() -> None:
    global _LAST_CALL_AT
    now = time.monotonic()
    wait = _MIN_INTERVAL - (now - _LAST_CALL_AT)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL_AT = time.monotonic()


def call_api(prompt: str, options: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    vars_ = context.get("vars") or {}
    thread_id = os.getenv("ASKPDF_EVAL_THREAD_ID", "").strip()
    if not thread_id:
        return {"error": "ASKPDF_EVAL_THREAD_ID not set; run via evals/promptfoo/run_eval.sh"}
    case = json.loads(vars_.get("case_json") or "{}") if vars_.get("case_json") else {}
    if not case:
        case = {
            "id": vars_.get("question_id"),
            "question": vars_.get("question") or prompt,
            "expected_pages": vars_.get("expected_pages") or [],
            "answer_match": vars_.get("answer_match") or {},
        }
    turn = {
        "question": vars_.get("question") or prompt,
        "expected_pages": vars_.get("expected_pages") or case.get("expected_pages") or [],
        "answer_match": case.get("answer_match") or {},
    }
    api_base = os.getenv("ASKPDF_BENCHMARK_API", os.getenv("ASKPDF_API_URL", "http://127.0.0.1:8000/api"))
    cfg = load_benchmark_config()
    thread_baseline = thread_settings_baseline(cfg)
    _throttle()
    try:
        with live_client(api_base=api_base) as client:
            row = run_turn(
                client,
                thread_id,
                case,
                turn,
                llm_model=CHAT_MODEL,
                judge=None,
                thread_baseline=thread_baseline,
                verify_settings=False,
            )
    except Exception as exc:
        return {"error": str(exc)}
    return {
        "output": row.get("answer") or row.get("answer_preview", ""),
        "metadata": row,
    }