#!/usr/bin/env python3
"""Cross-paper live check: BERT PDF + off-topic decline (not Attention-specific)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import live_attention_matrix as lam  # noqa: E402

MODEL = (lam.MODELS[0] if lam.MODELS else os.environ.get("ASKPDF_E2E_LLM_MODEL", "inclusionai/ling-3.2-flash:free"))
PDF = Path("/opt/cursor/artifacts/pdfs/bert.pdf")
DECLINE = lam.DECLINE

CASES = [
    (
        "router_rag_agent",
        "bert_architecture",
        "What model architecture does the BERT paper introduce?",
        lambda a: DECLINE not in a and ("transformer" in a.casefold() or "bidirectional" in a.casefold()),
    ),
    (
        "router_rag_agent",
        "bert_offtopic_paris",
        "What is the capital of France?",
        lambda a: a.strip() == DECLINE,
    ),
    (
        "corrective_self_rag_agent",
        "bert_offtopic_weather",
        "What is the weather in Seattle today?",
        lambda a: a.strip() == DECLINE,
    ),
]


def main() -> int:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if not PDF.is_file():
        raise FileNotFoundError(f"BERT PDF missing: {PDF}")
    lam.PDF = PDF
    project = lam.http_json("POST", "/api/projects", {"name": "cross-paper", "embedding_model": "BAAI/bge-m3"})
    thread = lam.http_json("POST", f"/api/projects/{project['id']}/threads", {"name": "bert"})
    thread_id = thread["id"]
    lam.upload_and_index(thread_id)
    rows = []
    for workflow, key, question, pred in CASES:
        lam.set_workflow(thread_id, workflow)
        ms, answer, body = lam.chat(thread_id, question, lam.runtime_model_id(MODEL))
        passed = bool(pred(answer))
        run_id = str(body.get("run_id") or body.get("agent_run_id") or "")
        scores = lam.export_trace_scores(thread_id, run_id) if run_id else []
        row = {
            "model": MODEL,
            "workflow": workflow,
            "scenario": key,
            "pass": passed,
            "answer": answer,
            "ms": ms,
            "has_raw_retriever_score": any(
                isinstance(s.get("raw_retriever_score"), (int, float)) for s in scores
            ),
        }
        rows.append(row)
        lam._write_screenshot(
            lam.ART / f"live-cross-{workflow}-{key}.png",
            workflow,
            key,
            question,
            answer,
            passed,
            MODEL,
        )
        print(json.dumps(row, indent=2), flush=True)
    out = {"sha": sha, "pdf": str(PDF), "rows": rows}
    (lam.ART / "live-cross-paper-matrix.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())