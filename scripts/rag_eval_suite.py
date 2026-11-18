#!/usr/bin/env python3
"""promptfoo-backed RAG eval suite (replaces benchmarks/run_benchmark.py)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from benchmark_config import context_baseline_record, load_benchmark_config, thread_settings_baseline  # noqa: E402
from benchmark_execution import run_case  # noqa: E402
from benchmark_fixtures import ensure_optional_arxiv, fixture_path, indexing_timeout_seconds  # noqa: E402
from benchmark_scoring import aggregate, default_judge_model, judge_request_timeout_s, llm_judge  # noqa: E402
from live_stack_client import admin_token, create_thread, live_client, upload_pdf, wait_indexed  # noqa: E402

QUESTIONS = REPO / "benchmarks" / "questions.json"
REPORTS = REPO / "benchmarks" / "reports"
SWEEPS = REPO / "evals" / "sweeps"
DEFAULT_API = os.getenv("ASKPDF_BENCHMARK_API", os.getenv("ASKPDF_API_URL", "http://127.0.0.1:8000/api"))
CHAT_MODEL = os.getenv("ASKPDF_BENCHMARK_MODEL", "nvidia/nemotron-3.5-super-vl:free")
FALLBACK_MODEL = os.getenv("ASKPDF_BENCHMARK_MODEL_FALLBACK", "particle-ai/particle-1:free")
JUDGE_MODEL = default_judge_model()
DEFAULT_WORKFLOWS = ("router_rag_agent", "auto_rag_agent")
MIN_INTERVAL = float(os.getenv("ASKPDF_EVAL_MIN_INTERVAL_S", "3.5"))


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()


def _throttle() -> None:
    time.sleep(MIN_INTERVAL)


def _load_cases(*, suite: str) -> list[dict[str, Any]]:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    out = []
    for case in data.get("questions") or []:
        suites = case.get("suites") or ["baseline"]
        if suite not in suites:
            continue
        out.append(case)
    return out


def _upload_fixtures(client: httpx.Client, thread_id: str, files: list[str]) -> None:
    for name in files:
        path = fixture_path(name)
        fh = upload_pdf(client, thread_id, path)
        wait_indexed(client, thread_id=thread_id, file_hash=fh, timeout=indexing_timeout_seconds(name))
        print(f"indexed {name}", flush=True)


def _prepare_optional_fixtures(files: list[str]) -> list[str]:
    ready = []
    for name in files:
        try:
            fixture_path(name)
            ready.append(name)
        except (FileNotFoundError, RuntimeError):
            try:
                ensure_optional_arxiv(name)
                ready.append(name)
            except Exception as exc:
                print(f"skip optional fixture {name}: {exc}", flush=True)
    return ready


def write_report(report: dict[str, Any], sha: str, label: str) -> tuple[Path, Path]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    stem = f"{sha[:12]}-{label}"
    json_path = REPORTS / f"{stem}.json"
    md_path = REPORTS / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        f"# RAG eval — `{label}` (`{sha[:12]}`)",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Chat model: `{report['chat_model']}`",
        f"- Judge model: `{report['judge_model']}`",
        "",
        "| Workflow | Cases | Retrieval | Answers | Grounding | Median latency (s) | Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for wf, agg in (report.get("workflows") or {}).items():
        lines.append(
            f"| `{wf}` | {agg['cases']} | {agg['retrieval_hit_rate']:.0%} | "
            f"{agg['answer_pass_rate']:.0%} | {agg.get('grounding_pass_rate', 0):.0%} | "
            f"{agg['median_latency_s']:.1f} | {agg['total_llm_tokens']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def run_suite(
    *,
    suite: str,
    workflows: list[str],
    limit: int,
    judge: bool,
    verify_settings: bool,
    config_path: str | None,
    label: str,
) -> dict[str, Any]:
    if config_path:
        os.environ["ASKPDF_BENCHMARK_CONFIG"] = config_path
    admin_token()
    benchmark_cfg = load_benchmark_config()
    thread_baseline = thread_settings_baseline(benchmark_cfg)
    cases = _load_cases(suite=suite)
    if limit > 0:
        cases = cases[:limit]
    optional_cases = [c for c in cases if c.get("optional_fixture")]
    required_files = sorted({f for c in cases if not c.get("optional_fixture") for f in c.get("fixture_files") or []})
    for c in optional_cases:
        required_files.extend(_prepare_optional_fixtures(c.get("fixture_files") or []))
    required_files = sorted(set(required_files))

    sha = _git_sha()
    report: dict[str, Any] = {
        "schema_version": 2,
        "commit_sha": sha,
        "label": label,
        "suite": suite,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chat_model": CHAT_MODEL,
        "judge_model": JUDGE_MODEL,
        "context_baseline": context_baseline_record(benchmark_cfg),
        "workflows": {},
        "cases": [],
    }

    judge_timeout = judge_request_timeout_s()
    judge_client = httpx.Client(
        timeout=httpx.Timeout(judge_timeout, connect=30.0),
        base_url="http://127.0.0.1:8000",
    )

    def judge_fn(q: str, a: str, m: dict[str, Any]) -> bool:
        _throttle()
        if judge:
            return llm_judge(judge_client, q, a, m, judge_model=JUDGE_MODEL)
        return False

    with live_client(api_base=DEFAULT_API) as client:
        for workflow in workflows:
            thread_id = create_thread(
                client,
                f"eval-{label}-{workflow}-{int(time.time())}",
                workflow_id=workflow,
                settings_patch=thread_baseline,
            )
            _upload_fixtures(client, thread_id, required_files)
            rows: list[dict[str, Any]] = []
            for case in cases:
                if case.get("optional_fixture") and not all(
                    f in required_files for f in (case.get("fixture_files") or [])
                ):
                    continue
                for row in run_case(
                    client,
                    thread_id,
                    case,
                    llm_model=CHAT_MODEL,
                    judge=judge_fn,
                    thread_baseline=thread_baseline,
                    verify_settings=verify_settings,
                ):
                    row["workflow"] = workflow
                    rows.append(row)
                    report["cases"].append(row)
                    print(json.dumps(row, default=str), flush=True)
                    _throttle()
                    if verify_settings and not row.get("settings_override_verified"):
                        raise RuntimeError(
                            f"settings override not verified for {row.get('question_id')}/{row.get('turn_id')}: "
                            f"{row.get('settings_verify_reason')}"
                        )
            report["workflows"][workflow] = aggregate(rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run askPDF RAG eval suite")
    parser.add_argument("--suite", default="baseline", help="Question suite tag (baseline, sweep_core)")
    parser.add_argument("--workflows", nargs="*", default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--verify-settings", action="store_true")
    parser.add_argument("--config", default=os.getenv("ASKPDF_BENCHMARK_CONFIG", str(REPO / "benchmarks/benchmark_config.eval.json")))
    parser.add_argument("--label", default="run")
    args = parser.parse_args()
    if args.suite == "sweep_core":
        args.verify_settings = True
    report = run_suite(
        suite=args.suite,
        workflows=args.workflows,
        limit=args.limit,
        judge=args.judge or os.getenv("ASKPDF_EVAL_JUDGE", "1") == "1",
        verify_settings=args.verify_settings,
        config_path=args.config,
        label=args.label,
    )
    json_path, md_path = write_report(report, report["commit_sha"], args.label)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())