#!/usr/bin/env python3
"""Compare overview-question behavior across chat models (context vs reasoning)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

from live_stack_client import (  # noqa: E402
    create_thread,
    live_client,
    scored_agent_run,
    thread_chat,
    upload_and_wait,
)
from overview_context_probe import probe_row  # noqa: E402

REPO = SCRIPTS.parent
PDF = REPO / "benchmarks" / "fixtures" / "pdfs" / "1706.03762.pdf"
QUESTION = os.getenv("ASKPDF_OVERVIEW_QUESTION", "What does this paper discuss?")
WORKFLOW = os.getenv("ASKPDF_OVERVIEW_WORKFLOW", "auto_rag_agent")
OUT = Path(os.getenv("ASKPDF_MODEL_COMPARE_OUT", "/opt/cursor/artifacts/overview-model-compare.json"))

MODELS = [
    m.strip()
    for m in os.getenv(
        "ASKPDF_OVERVIEW_MODELS",
        "openai/gpt-4.1-mini,google/gemini-2.5-flash,openrouter/free",
    ).split(",")
    if m.strip()
]


def main() -> int:
    report = {"question": QUESTION, "workflow": WORKFLOW, "pdf": PDF.name, "runs": []}
    with live_client() as client:
        thread_id = create_thread(client, f"overview-compare-{int(time.time())}", workflow_id=WORKFLOW)
        report["thread_id"] = thread_id
        upload_and_wait(client, thread_id, PDF)
        for model in MODELS:
            started = time.time()
            print(f"running {model}...", flush=True)
            try:
                body = thread_chat(client, thread_id, QUESTION, llm_model=model)
                run_id = str(body.get("agent_run_id") or "")
                detail = scored_agent_run(client, thread_id, run_id) if run_id else {}
                row = probe_row(
                    model=model,
                    question=QUESTION,
                    chat_body=body,
                    run_detail=detail,
                    latency_s=round(time.time() - started, 2),
                )
            except Exception as exc:  # noqa: BLE001
                row = {"model": model, "error": str(exc)}
            report["runs"].append(row)
            print(json.dumps(row, indent=2), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())