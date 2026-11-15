#!/usr/bin/env python3
"""Run the askPDF RAG benchmark against a live stack."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from benchmark_config import (  # noqa: E402
    chat_request_options,
    context_baseline_record,
    load_benchmark_config,
    thread_settings_baseline,
)
from benchmark_papers import ensure_all, fixture_path, indexing_timeout_seconds  # noqa: E402
from live_stack_client import (  # noqa: E402
    admin_token,
    answer_from_run,
    create_thread,
    get_agent_run,
    live_client,
    pages_from_document_sources,
    scored_agent_run,
    thread_chat,
    upload_pdf,
    wait_indexed,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).resolve().parent / "fixtures" / "manifest.json"
QUESTIONS = Path(__file__).resolve().parent / "questions.json"
HOLDOUT = Path(__file__).resolve().parent / "holdout_questions.json"
REPORTS = Path(__file__).resolve().parent / "reports"
BASELINE_JSON = Path(__file__).resolve().parent / "baseline.json"

DEFAULT_API = os.getenv("ASKPDF_BENCHMARK_API", os.getenv("ASKPDF_API_URL", "http://127.0.0.1:8000/api"))
CHAT_MODEL = os.getenv("ASKPDF_BENCHMARK_MODEL", "openai/gpt-4.1-mini")
JUDGE_MODEL = os.getenv("ASKPDF_BENCHMARK_JUDGE_MODEL", "google/gemini-2.5-flash")
DECLINE_RE = re.compile(
    r"attached documents do not|do not address|not (?:found|present|available) in the",
    re.I,
)

JUDGE_SYSTEM_PROMPT = """You grade RAG answers against reference constraints.
Reply with JSON only: {"pass": true|false, "reason": "..."}.
Pass when the answer satisfies the constraints described in the user message."""

BUILTIN_WORKFLOWS = (
    "auto_rag_agent",
    "router_rag_agent",
    "plan_execute_rag_agent",
    "orchestrator_worker_rag_agent",
    "corrective_self_rag_agent",
    "evaluator_replanner_rag_agent",
)

DEFAULT_WORKFLOWS = ("router_rag_agent", "auto_rag_agent")

CATEGORY_LABELS = {
    "overview": "overview",
    "fact": "facts",
    "table": "tables",
    "method": "method",
    "multi_doc": "multi_doc",
    "out_of_scope": "out_of_scope",
    "results": "results",
}


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()


def _load_fixture_hashes() -> dict[str, str]:
    path = REPORTS / ".fixture_hashes.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_fixture_hashes(mapping: dict[str, str]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / ".fixture_hashes.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")


def upload_fixtures(client: httpx.Client, thread_id: str, files: list[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in files:
        file_hash = upload_pdf(client, thread_id, fixture_path(name))
        wait_indexed(
            client,
            thread_id=thread_id,
            file_hash=file_hash,
            timeout=indexing_timeout_seconds(name),
        )
        hashes[name] = file_hash
        print(f"indexed {name}", flush=True)
    return hashes


def _metrics_tokens(detail: dict[str, Any]) -> int:
    metrics = {}
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


def numeric_score(answer: str, spec: dict[str, Any] | None) -> bool | None:
    if not spec:
        return None
    target = float(spec["target"])
    tol = float(spec.get("tolerance") or 0)
    found = [float(x) for x in re.findall(r"-?\d+\.?\d*", answer)]
    if not found:
        return False
    return any(abs(v - target) <= tol for v in found)


def string_constraints(answer: str, match: dict[str, Any]) -> bool | None:
    lowered = answer.lower()
    if match.get("decline"):
        return bool(DECLINE_RE.search(answer))
    if match.get("all_contains"):
        return all(sub.lower() in lowered for sub in match["all_contains"])
    if match.get("any_contains"):
        return any(sub.lower() in lowered for sub in match["any_contains"])
    return None


def llm_judge(client: httpx.Client, question: str, answer: str, match: dict[str, Any]) -> bool:
    user = json.dumps({"question": question, "answer": answer, "constraints": match})
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": 0,
    }
    # OpenAI-compatible route on rag-service
    r = client.post("http://127.0.0.1:8000/v1/chat/completions", json=payload, timeout=120.0)
    if r.status_code >= 400:
        return string_constraints(answer, match) is True
    text = r.json()["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(text[text.find("{") : text.rfind("}") + 1])
        return bool(parsed.get("pass"))
    except (json.JSONDecodeError, ValueError):
        return False


def run_question(
    client: httpx.Client,
    thread_id: str,
    case: dict[str, Any],
    *,
    judge: Callable[[str, str, dict[str, Any]], bool],
    chat_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.time()
    body = thread_chat(
        client,
        thread_id,
        case["question"],
        llm_model=CHAT_MODEL,
        chat_options=chat_options,
    )
    run_id = body.get("agent_run_id")
    detail = scored_agent_run(client, thread_id, str(run_id)) if run_id else {}
    answer = str(body.get("answer") or answer_from_run(detail) or "")
    pages = pages_from_document_sources(body.get("document_sources") or [])
    expected = set(case.get("expected_pages") or [])
    retrieval_hit = (not expected) or bool(pages & expected)
    match = case.get("answer_match") or {}
    numeric = numeric_score(answer, case.get("numeric"))
    constrained = string_constraints(answer, match)
    if constrained is not None:
        answer_pass = constrained
    elif numeric is not None:
        answer_pass = numeric
    else:
        answer_pass = judge(case["question"], answer, match)
    latency_s = round(time.time() - started, 3)
    decline_pass = answer_pass if case.get("type") == "out_of_scope" else None
    category = CATEGORY_LABELS.get(str(case.get("type") or ""), str(case.get("type") or "other"))
    return {
        "question_id": case["id"],
        "type": case.get("type"),
        "category": category,
        "answer_preview": answer[:500],
        "retrieval_pages": sorted(pages),
        "retrieval_hit": retrieval_hit,
        "answer_pass": answer_pass,
        "decline_correct": decline_pass,
        "latency_s": latency_s,
        "llm_tokens_total": _run_llm_tokens(client, thread_id, body, detail),
        "run_id": detail.get("id") or run_id,
        "workflow_id": detail.get("workflow_id"),
    }


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r[key] for r in rows if r.get(key) is not None]
    if not values:
        return None
    return round(sum(1 for v in values if v) / len(values), 4)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows) or 1
    latencies = sorted(r["latency_s"] for r in rows)
    by_category: dict[str, Any] = {}
    for category in sorted({r.get("category") for r in rows if r.get("category")}):
        subset = [r for r in rows if r.get("category") == category]
        by_category[category] = {
            "cases": len(subset),
            "retrieval_hit_rate": _rate(subset, "retrieval_hit"),
            "answer_pass_rate": _rate(subset, "answer_pass"),
            "decline_correct_rate": _rate(subset, "decline_correct"),
            "median_latency_s": sorted(r["latency_s"] for r in subset)[len(subset) // 2] if subset else 0,
            "total_llm_tokens": sum(int(r.get("llm_tokens_total") or 0) for r in subset),
        }
    return {
        "cases": len(rows),
        "retrieval_hit_rate": round(sum(1 for r in rows if r["retrieval_hit"]) / n, 4),
        "answer_pass_rate": round(sum(1 for r in rows if r["answer_pass"]) / n, 4),
        "decline_correct_rate": _rate(rows, "decline_correct"),
        "median_latency_s": latencies[len(latencies) // 2] if latencies else 0,
        "total_llm_tokens": sum(int(r.get("llm_tokens_total") or 0) for r in rows),
        "by_category": by_category,
    }


def print_baseline_deltas(report: dict[str, Any]) -> None:
    if not BASELINE_JSON.is_file():
        return
    baseline = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    ref = baseline.get("workflows") if isinstance(baseline.get("workflows"), dict) else {}
    ref_sha = str(baseline.get("commit_sha") or "")[:12]
    for workflow, agg in (report.get("workflows") or {}).items():
        block = ref.get(workflow)
        if not isinstance(block, dict):
            continue
        dr = float(agg["retrieval_hit_rate"]) - float(block["retrieval_hit_rate"])
        da = float(agg["answer_pass_rate"]) - float(block["answer_pass_rate"])
        print(
            f"vs baseline ({ref_sha}) {workflow}: retrieval {dr:+.0%}, answers {da:+.0%}",
            flush=True,
        )


def write_report(report: dict[str, Any], sha: str) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS / f"{sha}.json"
    md_path = REPORTS / f"{sha}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        f"# RAG benchmark — `{sha[:12]}`",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Chat model: `{report['chat_model']}`",
        f"- Judge model: `{report['judge_model']}`",
        "",
    ]
    baseline = report.get("context_baseline")
    if isinstance(baseline, dict):
        ctx = baseline.get("context") if isinstance(baseline.get("context"), dict) else {}
        lines += [
            "## Context baseline",
            "",
            f"- Config: `{baseline.get('config_path', 'benchmarks/benchmark_config.json')}`",
            f"- `context_window`: {ctx.get('context_window')}",
            f"- `use_reranker`: {ctx.get('use_reranker')}",
            f"- `use_web_search`: {ctx.get('use_web_search')}",
            "",
        ]
        budget = ctx.get("context_budget")
        if isinstance(budget, dict):
            lines.append("- `context_budget`: " + ", ".join(f"{k}={v}" for k, v in sorted(budget.items())))
            lines.append("")
    lines += [
        "## Summary by workflow",
        "",
        "| Workflow | Cases | Retrieval | Answers | Median latency (s) | LLM tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for wf, agg in report.get("workflows", {}).items():
        lines.append(
            f"| `{wf}` | {agg['cases']} | {agg['retrieval_hit_rate']:.0%} | "
            f"{agg['answer_pass_rate']:.0%} | {agg['median_latency_s']:.1f} | {agg['total_llm_tokens']} |"
        )
        lines.append("")
        lines.append(f"### `{wf}` by category")
        lines.append("")
        lines.append("| Category | Cases | Retrieval | Answers | Decline | Latency (med s) | Tokens |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for cat, block in (agg.get("by_category") or {}).items():
            decl = block.get("decline_correct_rate")
            decl_s = f"{decl:.0%}" if decl is not None else "—"
            lines.append(
                f"| {cat} | {block['cases']} | {block['retrieval_hit_rate']:.0%} | "
                f"{block['answer_pass_rate']:.0%} | {decl_s} | {block['median_latency_s']:.1f} | {block['total_llm_tokens']} |"
            )
        lines.append("")
    if report.get("holdout"):
        h = report["holdout"]["aggregate"]
        lines += [
            "",
            "## Holdout (do not tune)",
            "",
            f"- Retrieval: {h['retrieval_hit_rate']:.0%}",
            f"- Answers: {h['answer_pass_rate']:.0%}",
        ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run askPDF RAG benchmark")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--workflows", nargs="*", default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--all-workflows", action="store_true")
    parser.add_argument("--holdout", action="store_true", help="Include holdout split")
    parser.add_argument("--holdout-only", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Limit main questions (smoke)")
    parser.add_argument("--judge", action="store_true", help="Use LLM judge when constraints are inconclusive")
    parser.add_argument("--reuse-threads", action="store_true")
    args = parser.parse_args()
    workflows = list(BUILTIN_WORKFLOWS) if args.all_workflows else args.workflows
    admin_token()  # validate before long run
    benchmark_cfg = load_benchmark_config()
    chat_opts = chat_request_options(benchmark_cfg)
    thread_baseline = thread_settings_baseline(benchmark_cfg)
    ensure_all(include_holdout=args.holdout or args.holdout_only)
    sha = _git_sha()
    main_cases = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if args.limit > 0:
        main_cases = main_cases[: args.limit]
    holdout_cases = json.loads(HOLDOUT.read_text(encoding="utf-8"))["questions"] if args.holdout or args.holdout_only else []

    report: dict[str, Any] = {
        "schema_version": 1,
        "commit_sha": sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chat_model": CHAT_MODEL,
        "judge_model": JUDGE_MODEL,
        "context_baseline": context_baseline_record(benchmark_cfg),
        "workflows": {},
        "cases": [],
    }

    with live_client(api_base=args.api) as client:
        judge_client = httpx.Client(timeout=120.0)

        def judge_fn(q: str, a: str, m: dict[str, Any]) -> bool:
            if args.judge:
                return llm_judge(judge_client, q, a, m)
            return False

        bench_files = sorted({f for c in main_cases for f in c.get("fixture_files", [])})
        holdout_files = sorted({f for c in holdout_cases for f in c.get("fixture_files", [])})
        all_main_set = sorted(set(bench_files))

        for workflow in workflows:
            if args.holdout_only:
                continue
            stamp = int(time.time())
            thread_id = create_thread(
                client,
                f"benchmark-{workflow}-{stamp}",
                workflow_id=workflow,
                settings_patch=thread_baseline,
            )
            hashes = upload_fixtures(client, thread_id, all_main_set)
            _save_fixture_hashes(hashes)
            rows: list[dict[str, Any]] = []
            for case in main_cases:
                case_files = case.get("fixture_files") or []
                if not all(name in all_main_set for name in case_files):
                    continue
                row = run_question(client, thread_id, case, judge=judge_fn, chat_options=chat_opts)
                row["workflow"] = workflow
                rows.append(row)
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
            report["workflows"][workflow] = aggregate(rows)

        if holdout_cases and (args.holdout or args.holdout_only):
            wf = "router_rag_agent"
            thread_id = create_thread(
                client,
                f"benchmark-holdout-{int(time.time())}",
                workflow_id=wf,
                settings_patch=thread_baseline,
            )
            needed = sorted(set(holdout_files))
            upload_fixtures(client, thread_id, needed)
            hrows = []
            for case in holdout_cases:
                row = run_question(client, thread_id, case, judge=judge_fn, chat_options=chat_opts)
                row["workflow"] = wf
                row["holdout"] = True
                hrows.append(row)
                report["cases"].append(row)
            report["holdout"] = {"aggregate": aggregate(hrows), "cases": hrows}

    json_path, md_path = write_report(report, sha)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print_baseline_deltas(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())