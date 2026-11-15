#!/usr/bin/env python3
"""Finish reproof matrix cases not completed before timeout."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import live_attention_matrix as lam  # noqa: E402

MODEL = lam.MODELS[0]
TAIL = [
    ("corrective_self_rag_agent", "bleu_en_de"),
    ("corrective_self_rag_agent", "paper_title"),
    ("corrective_self_rag_agent", "encoder_natural"),
    ("corrective_self_rag_agent", "tokyo"),
    ("router_rag_agent", "architecture"),
]


def main() -> int:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    print(f"SHA={sha} MODEL={MODEL}", flush=True)
    lam.wait_langgraph_ready()
    project = lam.http_json(
        "POST",
        "/api/projects",
        {"name": "live-reproof-tail", "description": "attention", "embedding_model": "BAAI/bge-m3"},
    )
    thread = lam.http_json("POST", f"/api/projects/{project['id']}/threads", {"name": "tail"})
    thread_id = thread["id"]
    lam.upload_and_index(thread_id)
    rows: list[dict] = []
    for workflow, scenario in TAIL:
        lam.set_workflow(thread_id, workflow)
        extra = {k: (k, q, p) for k, q, p in lam.EXTRA_SCENARIOS_BY_WORKFLOW.get(workflow, [])}
        base = {k: (k, q, p) for k, q, p in lam.SCENARIOS}
        key, question, pred = extra.get(scenario) or base[scenario]
        print(f"RUN {workflow}/{scenario} ...", flush=True)
        ms, answer, body = lam.chat(thread_id, question, MODEL)
        passed = bool(pred(answer))
        row = {
            "workflow": workflow,
            "scenario": scenario,
            "pass": passed,
            "ms": ms,
            "answer": answer,
        }
        rows.append(row)
        print(json.dumps({**row, "answer": answer[:300]}), flush=True)
    out = {"sha": sha, "model": MODEL, "rows": rows}
    (lam.ART / "live-reproof-tail.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())