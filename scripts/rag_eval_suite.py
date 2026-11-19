#!/usr/bin/env python3
"""promptfoo-backed RAG eval suite (replaces benchmarks/run_benchmark.py)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

SCRIPTS = Path(__file__).resolve().parent
REPO = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

from benchmark_config import context_baseline_record, load_benchmark_config, thread_settings_baseline  # noqa: E402
from benchmark_eval_support import (  # noqa: E402
    EvalProfile,
    RequestPacer,
    apply_reasoning_effort,
    bundled_fixture_names,
    case_result_key,
    checkpoint_path,
    load_cases,
    load_checkpoint,
    resolve_chat_models,
    resolve_profile,
    save_checkpoint,
    stack_api_base,
    wait_stack_ready,
)
from benchmark_execution import run_case  # noqa: E402
from benchmark_fixtures import fixture_path, indexing_timeout_seconds  # noqa: E402
from benchmark_scoring import aggregate, default_judge_model, judge_request_timeout_s, llm_judge  # noqa: E402
from live_stack_client import admin_token, create_thread, live_client, upload_pdf, wait_indexed  # noqa: E402

REPORTS = REPO / "benchmarks" / "reports"
DEFAULT_WORKFLOWS = ("router_rag_agent", "auto_rag_agent")
JUDGE_MODEL = default_judge_model()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()


def _case_keys(case: dict[str, Any], workflow: str) -> list[str]:
    turns = case.get("turns")
    if isinstance(turns, list) and turns:
        return [case_result_key(str(case.get("id")), str(t.get("id")), workflow) for t in turns]
    return [case_result_key(str(case.get("id")), None, workflow)]


def _upload_fixtures(
    client: httpx.Client,
    thread_id: str,
    files: list[str],
    *,
    pacer: RequestPacer,
) -> set[str]:
    ready: set[str] = set()
    for name in files:
        path = fixture_path(name)

        def _upload_wait() -> str:
            fh = upload_pdf(client, thread_id, path)
            wait_indexed(client, thread_id=thread_id, file_hash=fh, timeout=indexing_timeout_seconds(name))
            return fh

        pacer.call(_upload_wait)
        print(f"indexed {name}", flush=True)
        ready.add(name)
    return ready


def _ensure_case_fixtures(
    client: httpx.Client,
    thread_id: str,
    case: dict[str, Any],
    *,
    uploaded: set[str],
    pacer: RequestPacer,
) -> None:
    for name in case.get("fixture_files") or []:
        if name in uploaded:
            continue
        path = fixture_path(name)

        def _upload_wait() -> None:
            fh = upload_pdf(client, thread_id, path)
            wait_indexed(client, thread_id=thread_id, file_hash=fh, timeout=indexing_timeout_seconds(name))

        pacer.call(_upload_wait)
        print(f"indexed {name}", flush=True)
        uploaded.add(name)


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
        f"- Profile: `{report.get('profile')}`",
        f"- Tier: `{report.get('tier')}`",
        f"- Chat model: `{report['chat_model']}`",
        f"- Judge model: `{report['judge_model']}`",
        f"- Reasoning effort: `{report.get('reasoning_effort', 'default')}`",
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
    comparison = report.get("reasoning_comparison")
    if isinstance(comparison, dict) and comparison:
        lines.extend(["", "## Reasoning comparison", ""])
        for label, block in comparison.items():
            lines.append(f"### effort={label}")
            for wf, agg in (block.get("workflows") or {}).items():
                lines.append(
                    f"- `{wf}`: answer_pass={agg.get('answer_pass_rate', 0):.0%}, "
                    f"median_latency={agg.get('median_latency_s', 0):.1f}s, tokens={agg.get('total_llm_tokens', 0)}"
                )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _run_single_case(
    *,
    api_base: str,
    workflow: str,
    case: dict[str, Any],
    pacer: RequestPacer,
    chat_model: str,
    judge_fn: Callable[[str, str, dict[str, Any]], bool] | None,
    thread_baseline: dict[str, Any],
    verify_settings: bool,
) -> list[dict[str, Any]]:
    with live_client(api_base=api_base) as client:
        thread_id = create_thread(
            client,
            f"eval-case-{case.get('id')}-{workflow}-{int(time.time())}",
            workflow_id=workflow,
            settings_patch=thread_baseline,
        )
        for name in case.get("fixture_files") or []:

            def _upload_wait(n: str = name) -> None:
                path = fixture_path(n)
                fh = upload_pdf(client, thread_id, path)
                wait_indexed(client, thread_id=thread_id, file_hash=fh, timeout=indexing_timeout_seconds(n))

            pacer.call(_upload_wait)

        def _execute() -> list[dict[str, Any]]:
            return run_case(
                client,
                thread_id,
                case,
                llm_model=chat_model,
                judge=judge_fn,
                thread_baseline=thread_baseline,
                verify_settings=verify_settings,
            )

        return pacer.call(_execute)


def _run_cases_for_workflow(
    *,
    client: httpx.Client,
    api_base: str,
    workflow: str,
    cases: list[dict[str, Any]],
    required_files: list[str],
    profile: EvalProfile,
    pacer: RequestPacer,
    chat_model: str,
    judge_fn: Callable[[str, str, dict[str, Any]], bool] | None,
    thread_baseline: dict[str, Any],
    verify_settings: bool,
    label: str,
    completed: set[str],
    ckpt_path: Path,
    report_cases: list[dict[str, Any]],
    checkpoint_resume: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    thread_id = create_thread(
        client,
        f"eval-{label}-{workflow}-{int(time.time())}",
        workflow_id=workflow,
        settings_patch=thread_baseline,
    )

    def _record_rows(case_rows: list[dict[str, Any]]) -> None:
        for row in case_rows:
            key = case_result_key(str(row.get("question_id")), row.get("turn_id"), workflow)
            if key in completed:
                continue
            row["workflow"] = workflow
            rows.append(row)
            report_cases.append(row)
            completed.add(key)
            save_checkpoint(ckpt_path, completed, meta={"label": label, "workflow": workflow})
            print(json.dumps(row, default=str), flush=True)
            if verify_settings and not row.get("settings_override_verified"):
                raise RuntimeError(
                    f"settings override not verified for {row.get('question_id')}/{row.get('turn_id')}: "
                    f"{row.get('settings_verify_reason')}"
                )

    conversation = [c for c in cases if isinstance(c.get("turns"), list) and c.get("turns")]
    single_turn = [c for c in cases if c not in conversation]

    if profile.parallel and single_turn:
        pending = [
            c
            for c in single_turn
            if not (checkpoint_resume and all(k in completed for k in _case_keys(c, workflow)))
        ]
        if pending:
            with ThreadPoolExecutor(max_workers=profile.max_workers) as pool:
                futures = {
                    pool.submit(
                        _run_single_case,
                        api_base=api_base,
                        workflow=workflow,
                        case=case,
                        pacer=pacer,
                        chat_model=chat_model,
                        judge_fn=judge_fn,
                        thread_baseline=thread_baseline,
                        verify_settings=verify_settings,
                    ): case
                    for case in pending
                }
                for fut in as_completed(futures):
                    _record_rows(fut.result())
        single_turn = []

    if single_turn or conversation:
        uploaded = _upload_fixtures(client, thread_id, required_files, pacer=pacer)
        for case in [*single_turn, *conversation]:
            if checkpoint_resume and all(k in completed for k in _case_keys(case, workflow)):
                print(f"skip completed case {case.get('id')} ({workflow})", flush=True)
                continue
            _ensure_case_fixtures(client, thread_id, case, uploaded=uploaded, pacer=pacer)

            def _execute_case(c: dict[str, Any] = case) -> list[dict[str, Any]]:
                return run_case(
                    client,
                    thread_id,
                    c,
                    llm_model=chat_model,
                    judge=judge_fn,
                    thread_baseline=thread_baseline,
                    verify_settings=verify_settings,
                )

            _record_rows(pacer.call(_execute_case))
    return rows


def run_suite(
    *,
    tier: str,
    suite: str | None,
    workflows: list[str],
    limit: int,
    judge: bool,
    verify_settings: bool,
    config_path: str | None,
    label: str,
    profile: EvalProfile,
    pacer: RequestPacer,
    reasoning_effort: str | None,
    checkpoint_resume: bool,
) -> dict[str, Any]:
    if config_path:
        os.environ["ASKPDF_BENCHMARK_CONFIG"] = config_path
    os.environ.setdefault("ASKPDF_EVAL_REASONING_EFFORT", "none")
    apply_reasoning_effort(reasoning_effort or os.getenv("ASKPDF_EVAL_REASONING_EFFORT", "none"))

    wait_stack_ready()
    admin_token()
    benchmark_cfg = load_benchmark_config()
    thread_baseline = thread_settings_baseline(benchmark_cfg)
    cases = load_cases(tier=tier, suite=suite, limit=limit)
    if not cases:
        raise RuntimeError(f"no cases for tier={tier!r} suite={suite!r}")
    required_files = bundled_fixture_names(cases)
    chat_model, _fallback = resolve_chat_models()
    ckpt_path = checkpoint_path(label)
    completed = load_checkpoint(ckpt_path) if checkpoint_resume else set()
    if resume and completed:
        print(f"resuming from checkpoint with {len(completed)} completed keys", flush=True)

    sha = _git_sha()
    report: dict[str, Any] = {
        "schema_version": 3,
        "commit_sha": sha,
        "label": label,
        "tier": tier,
        "suite": suite,
        "profile": profile.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chat_model": chat_model,
        "judge_model": JUDGE_MODEL,
        "reasoning_effort": reasoning_effort or os.getenv("ASKPDF_EVAL_REASONING_EFFORT", "none"),
        "context_baseline": context_baseline_record(benchmark_cfg),
        "workflows": {},
        "cases": [],
    }

    judge_timeout = judge_request_timeout_s()
    api_root = stack_api_base().replace("/api", "")
    judge_client = httpx.Client(timeout=httpx.Timeout(judge_timeout, connect=30.0), base_url=api_root)

    def judge_fn(q: str, a: str, m: dict[str, Any]) -> bool:
        if judge:

            def _judge() -> bool:
                return llm_judge(judge_client, q, a, m, judge_model=JUDGE_MODEL)

            return bool(pacer.call(_judge))
        return False

    api_base = stack_api_base()
    with live_client(api_base=api_base) as client:
        for workflow in workflows:
            rows = _run_cases_for_workflow(
                client=client,
                api_base=api_base,
                workflow=workflow,
                profile=profile,
                cases=cases,
                required_files=required_files,
                pacer=pacer,
                chat_model=chat_model,
                judge_fn=judge_fn if judge else None,
                thread_baseline=thread_baseline,
                verify_settings=verify_settings,
                label=label,
                completed=completed,
                ckpt_path=ckpt_path,
                report_cases=report["cases"],
                checkpoint_resume=checkpoint_resume,
            )
            report["workflows"][workflow] = aggregate(rows)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run askPDF RAG eval suite")
    parser.add_argument("--tier", default=os.getenv("ASKPDF_EVAL_TIER", "core"), choices=("smoke", "core", "advanced", "holdout", "all"))
    parser.add_argument("--suite", default=None, help="Legacy suite tag filter (e.g. sweep_core)")
    parser.add_argument("--profile", default=None, choices=("fast", "free"), help="Throughput profile (env ASKPDF_EVAL_PROFILE)")
    parser.add_argument("--workflows", nargs="*", default=list(DEFAULT_WORKFLOWS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--verify-settings", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and rerun all cases")
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("ASKPDF_EVAL_REASONING_EFFORT", "none"),
        help="OpenRouter reasoning.effort for chat (default none)",
    )
    parser.add_argument(
        "--reasoning-compare",
        action="store_true",
        help="Run core+advanced twice with reasoning effort none vs ASKPDF_EVAL_REASONING_COMPARE_EFFORT (default high)",
    )
    parser.add_argument("--config", default=os.getenv("ASKPDF_BENCHMARK_CONFIG", str(REPO / "benchmarks/benchmark_config.eval.json")))
    parser.add_argument("--label", default="run")
    args = parser.parse_args()
    if args.suite == "sweep_core" or (args.suite is None and os.getenv("ASKPDF_EVAL_SUITE") == "sweep_core"):
        args.suite = "sweep_core"
        args.verify_settings = True
        if args.tier == "core":
            args.tier = "core"

    use_judge = args.judge or (not args.no_judge and os.getenv("ASKPDF_EVAL_JUDGE", "1") == "1")
    profile = resolve_profile(args.profile)
    pacer = RequestPacer(profile)

    def _single_run(tier: str, effort: str, suffix: str) -> dict[str, Any]:
        return run_suite(
            tier=tier,
            suite=args.suite,
            workflows=args.workflows,
            limit=args.limit,
            judge=use_judge,
            verify_settings=args.verify_settings,
            config_path=args.config,
            label=f"{args.label}-{suffix}" if suffix else args.label,
            profile=profile,
            pacer=pacer,
            checkpoint_resume=not args.no_resume,
            reasoning_effort=effort,
        )

    if args.reasoning_compare:
        if args.tier not in {"core", "advanced", "all"}:
            raise SystemExit("--reasoning-compare requires --tier core, advanced, or all")
        compare_effort = os.getenv("ASKPDF_EVAL_REASONING_COMPARE_EFFORT", "high")
        report = _single_run(args.tier, "none", "reasoning-none")
        report["reasoning_comparison"] = {"none": {"workflows": report["workflows"]}}
        second = _single_run(args.tier, compare_effort, "reasoning-on")
        report["reasoning_comparison"][compare_effort] = {"workflows": second["workflows"]}
        report["cases"].extend(second["cases"])
    else:
        report = _single_run(args.tier, args.reasoning_effort, "")

    json_path, md_path = write_report(report, report["commit_sha"], args.label)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())