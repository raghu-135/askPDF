"""Execute benchmark cases against a live stack (single- and multi-turn)."""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Callable

import httpx

from benchmark_config import chat_request_options, load_benchmark_config
from benchmark_effective_settings import assert_override_applied, effective_settings_from_run
from benchmark_scoring import CATEGORY_LABELS, score_answer
from live_stack_client import (
    answer_from_run,
    get_agent_run,
    pages_from_document_sources,
    scored_agent_run,
    thread_chat,
    update_thread_settings,
)


def _metrics_tokens(detail: dict[str, Any]) -> int:
    metrics: dict[str, Any] = {}
    debug = detail.get("debug") if isinstance(detail.get("debug"), dict) else {}
    trace = debug.get("trace") if isinstance(debug.get("trace"), dict) else {}
    if isinstance(trace.get("metrics"), dict):
        metrics = trace["metrics"]
    elif isinstance(detail.get("metrics_json"), dict):
        metrics = detail["metrics_json"]
    return int(metrics.get("llm_token_count_total") or 0)


def _run_llm_tokens(client: httpx.Client, thread_id: str, chat_body: dict[str, Any], detail: dict[str, Any]) -> int:
    total = _metrics_tokens(detail)
    parent_id = chat_body.get("agent_run_id")
    if parent_id and str(detail.get("id")) != str(parent_id):
        total += _metrics_tokens(get_agent_run(client, thread_id, str(parent_id)))
    return total


def merge_case_chat_options(case: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    base = chat_request_options(config)
    override = case.get("chat_options") if isinstance(case.get("chat_options"), dict) else {}
    out = deepcopy(base)
    for key, value in override.items():
        if key == "context_budget" and isinstance(value, dict):
            merged_budget = dict(out.get("context_budget") or {})
            merged_budget.update(value)
            out["context_budget"] = merged_budget
        else:
            out[key] = value
    return out


def merge_case_thread_settings(case: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    patch = case.get("thread_settings") if isinstance(case.get("thread_settings"), dict) else {}
    out = deepcopy(baseline)
    for key, value in patch.items():
        if key == "context_budget" and isinstance(value, dict):
            merged = dict(out.get("context_budget") or {})
            merged.update(value)
            out["context_budget"] = merged
        elif key == "coverage_iterations" and isinstance(value, dict):
            merged = dict(out.get("coverage_iterations") or {})
            merged.update(value)
            out["coverage_iterations"] = merged
        else:
            out[key] = value
    for flat in (
        "max_iterations_sufficient_coverage",
        "max_iterations_probably_sufficient_coverage",
        "web_search_iteration_bonus",
    ):
        if flat in patch:
            merged = dict(out.get("coverage_iterations") or {})
            merged[flat] = patch[flat]
            out["coverage_iterations"] = merged
    return out


def expected_effective_settings(case: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    merged = merge_case_thread_settings(case, baseline)
    opts = merge_case_chat_options(case, load_benchmark_config())
    out: dict[str, Any] = {}
    if opts.get("context_window") is not None:
        out["context_window"] = opts["context_window"]
    elif merged.get("context_window") is not None:
        out["context_window"] = merged["context_window"]
    if isinstance(opts.get("context_budget"), dict):
        out["context_budget"] = opts["context_budget"]
    elif isinstance(merged.get("context_budget"), dict) and merged["context_budget"]:
        out["context_budget"] = merged["context_budget"]
    cov = merged.get("coverage_iterations")
    if isinstance(cov, dict) and cov:
        out["coverage_iterations"] = cov
    return out


def run_turn(
    client: httpx.Client,
    thread_id: str,
    case: dict[str, Any],
    turn: dict[str, Any],
    *,
    llm_model: str,
    judge: Callable[[str, str, dict[str, Any]], bool] | None,
    thread_baseline: dict[str, Any],
    verify_settings: bool = False,
) -> dict[str, Any]:
    chat_opts = merge_case_chat_options(case)
    ts_patch = merge_case_thread_settings(case, thread_baseline)
    if ts_patch:
        update_thread_settings(client, thread_id, ts_patch)
    started = time.time()
    body = thread_chat(
        client,
        thread_id,
        str(turn.get("question") or case.get("question") or ""),
        llm_model=llm_model,
        bypass_clarification=bool(turn.get("bypass_clarification", case.get("bypass_clarification", True))),
        chat_options=chat_opts,
    )
    run_id = body.get("agent_run_id")
    detail = scored_agent_run(client, thread_id, str(run_id)) if run_id else {}
    answer = str(body.get("answer") or answer_from_run(detail) or "")
    pages = pages_from_document_sources(body.get("document_sources") or [])
    expected = set(turn.get("expected_pages") or case.get("expected_pages") or [])
    retrieval_hit = (not expected) or bool(pages & expected)
    scoring_case = {**case, **turn}
    scores = score_answer(str(turn.get("question") or ""), answer, scoring_case, judge=judge)
    effective = effective_settings_from_run(detail)
    settings_ok, settings_reason = True, "ok"
    if verify_settings:
        settings_ok, settings_reason = assert_override_applied(
            expected_effective_settings(case, thread_baseline),
            effective,
        )
    latency_s = round(time.time() - started, 3)
    return {
        "question_id": case.get("id"),
        "turn_id": turn.get("id"),
        "type": case.get("type"),
        "category": CATEGORY_LABELS.get(str(case.get("type") or ""), str(case.get("type") or "other")),
        "answer": answer,
        "answer_preview": answer[:500],
        "retrieval_pages": sorted(pages),
        "retrieval_hit": retrieval_hit,
        "latency_s": latency_s,
        "llm_tokens_total": _run_llm_tokens(client, thread_id, body, detail),
        "run_id": detail.get("id") or run_id,
        "effective_settings": effective,
        "settings_override_verified": settings_ok,
        "settings_verify_reason": settings_reason,
        **scores,
        "decline_correct": scores["answer_pass"] if case.get("type") == "out_of_scope" else None,
    }


def run_case(
    client: httpx.Client,
    thread_id: str,
    case: dict[str, Any],
    *,
    llm_model: str,
    judge: Callable[[str, str, dict[str, Any]], bool] | None,
    thread_baseline: dict[str, Any],
    verify_settings: bool = False,
) -> list[dict[str, Any]]:
    turns = case.get("turns")
    if isinstance(turns, list) and turns:
        rows = []
        for turn in turns:
            rows.append(
                run_turn(
                    client,
                    thread_id,
                    case,
                    turn,
                    llm_model=llm_model,
                    judge=judge,
                    thread_baseline=thread_baseline,
                    verify_settings=verify_settings,
                )
            )
        return rows
    return [
        run_turn(
            client,
            thread_id,
            case,
            case,
            llm_model=llm_model,
            judge=judge,
            thread_baseline=thread_baseline,
            verify_settings=verify_settings,
        )
    ]