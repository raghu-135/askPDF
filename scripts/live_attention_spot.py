#!/usr/bin/env python3
"""Run a subset of live_attention_matrix cases for fast iteration."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import live_attention_matrix as lam  # noqa: E402


SPOTS = [
    ("router_rag_agent", "architecture"),
    ("corrective_self_rag_agent", "paper_title"),
    ("corrective_self_rag_agent", "tokyo"),
    ("orchestrator_worker_rag_agent", "tokyo"),
]


def main() -> int:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    project = lam.http_json("POST", "/api/projects", {"name": "live-spot", "description": "attention", "embedding_model": "BAAI/bge-m3"})
    thread = lam.http_json("POST", f"/api/projects/{project['id']}/threads", {"name": "spot"})
    thread_id = thread["id"]
    lam.upload_and_index(thread_id)

    rows: list[dict] = []
    for workflow, scenario in SPOTS:
        lam.set_workflow(thread_id, workflow)
        extra = {key: (key, q, p) for key, q, p in lam.EXTRA_SCENARIOS_BY_WORKFLOW.get(workflow, [])}
        base = {key: (key, q, p) for key, q, p in lam.SCENARIOS}
        key, question, pred = extra.get(scenario) or base[scenario]
        ms, answer, body = lam.chat(thread_id, question)
        run_id = str(body.get("run_id") or body.get("agent_run_id") or "")
        scores = lam.export_trace_scores(thread_id, run_id) if run_id else []
        passed = bool(pred(answer))
        row = {
            "workflow": workflow,
            "scenario": scenario,
            "pass": passed,
            "ms": ms,
            "answer": answer,
            "run_id": run_id,
            "trace_score_packets": len(scores),
            "has_raw_retriever_score": any(
                isinstance(s.get("raw_retriever_score"), (int, float)) for s in scores
            ),
        }
        rows.append(row)
        lam._write_screenshot(
            lam.ART / f"live-{workflow}-{scenario}.png",
            workflow,
            scenario,
            question,
            answer,
            passed,
        )
        print(json.dumps(row, indent=2))

    out = {"sha": sha, "rows": rows, "thread_id": thread_id}
    (lam.ART / "live-attention-spot.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())