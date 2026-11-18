#!/usr/bin/env python3
"""Fast nemotron spot check for encoder hedge + Tokyo decline (user report)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import live_attention_matrix as lam  # noqa: E402

MODEL = "nvidia/nemotron-3.5-super"
RUNTIME_MODEL = lam.runtime_model_id(MODEL)
CASES = [
    ("router_rag_agent", "encoder_natural"),
    ("router_rag_agent", "tokyo"),
    ("corrective_self_rag_agent", "encoder_natural"),
    ("corrective_self_rag_agent", "tokyo"),
]


def lookup(workflow: str, key: str):
    for key_i, question, pred in [*lam.SCENARIOS, *lam.EXTRA_SCENARIOS_BY_WORKFLOW.get(workflow, [])]:
        if key_i == key:
            return question, pred
    raise KeyError((workflow, key))


def main() -> int:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    project = lam.http_json(
        "POST",
        "/api/projects",
        {"name": "nemotron-spot", "description": "attention", "embedding_model": "BAAI/bge-m3"},
    )
    thread = lam.http_json("POST", f"/api/projects/{project['id']}/threads", {"name": "nemotron-spot"})
    thread_id = thread["id"]
    lam.upload_and_index(thread_id)
    rows = []
    for workflow, key in CASES:
        lam.set_workflow(thread_id, workflow)
        question, pred = lookup(workflow, key)
        ms, answer, body = lam.chat(thread_id, question, RUNTIME_MODEL)
        passed = bool(pred(answer))
        row = {
            "model": MODEL,
            "runtime_model": RUNTIME_MODEL,
            "workflow": workflow,
            "scenario": key,
            "pass": passed,
            "ms": ms,
            "answer": answer,
            "sha": sha,
        }
        rows.append(row)
        lam._write_screenshot(
            lam.ART / f"live-nemotron-{workflow}-{key}.png",
            workflow,
            key,
            question,
            answer,
            passed,
            MODEL,
        )
        print(json.dumps(row, indent=2), flush=True)
    out = {"sha": sha, "model": MODEL, "rows": rows}
    (lam.ART / "live-nemotron-spot.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())