#!/usr/bin/env python3
"""Run Auto orchestrator live cases a,b,c,d,f against local stack."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx

from live_stack_client import (
    create_thread,
    get_agent_run,
    live_client,
    thread_chat,
    upload_pdf,
    wait_indexed,
)

MODEL = "openai/gpt-4.1-mini"
ATTENTION = Path("/opt/cursor/artifacts/pdfs/attention.pdf")
BERT = Path("/opt/cursor/artifacts/pdfs/bert.pdf")


def wait_ready(client: httpx.Client, thread_id: str, file_hash: str, timeout: float = 600.0) -> bool:
    try:
        wait_indexed(client, thread_id=thread_id, file_hash=file_hash, timeout=timeout)
        return True
    except (TimeoutError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return False


def resume_run(client: httpx.Client, thread_id: str, run_id: str, interrupt_id: str) -> dict:
    r = client.post(
        f"/agent-runs/{run_id}/resume",
        params={"thread_id": thread_id},
        json={"interrupt_id": interrupt_id, "action": "approve"},
        timeout=600.0,
    )
    r.raise_for_status()
    return r.json()


def chat(client: httpx.Client, thread_id: str, question: str) -> dict:
    return thread_chat(client, thread_id, question, llm_model=MODEL)


def run_debug(client: httpx.Client, thread_id: str, run_id: str) -> dict:
    return get_agent_run(client, thread_id, run_id)


def main() -> int:
    results: list[dict] = []
    with live_client() as client:
        thread_id = create_thread(client, f"auto-live-{int(time.time())}")
        print(f"thread_id={thread_id}")
        att_hash = upload_pdf(client, thread_id, ATTENTION)
        bert_hash = upload_pdf(client, thread_id, BERT)
        print("waiting for indexing...")
        if not wait_ready(client, thread_id, att_hash) or not wait_ready(client, thread_id, bert_hash):
            print("indexing timeout", file=sys.stderr)
            return 1

        cases = [
            ("a", "What is the scaled dot-product attention mechanism described in the Attention paper?", []),
            ("c", "What is the population of Tokyo?", []),
            ("d", "Write a literature review comparing transformer attention with BERT's bidirectional encoding.", []),
            (
                "b",
                "Compare the Attention paper and the BERT paper on architecture and training objectives.",
                ["approve"],
            ),
        ]
        for case_id, question, actions in cases:
            out = chat(client, thread_id, question)
            run_id = out.get("agent_run_id")
            status = out.get("status")
            answer = str(out.get("answer") or out.get("final_answer") or "")
            interrupt = out.get("pending_interrupt") or {}
            if actions and status == "awaiting_human" and interrupt.get("interrupt_id") and run_id:
                resumed = resume_run(client, thread_id, run_id, str(interrupt["interrupt_id"]))
                status = resumed.get("status") or (resumed.get("agent_run") or {}).get("status")
                answer = str(
                    resumed.get("answer")
                    or resumed.get("final_answer")
                    or (resumed.get("agent_run") or {}).get("final_output", {}).get("answer")
                    or ""
                )
            debug = run_debug(client, thread_id, run_id) if run_id else {}
            tech = (debug.get("debug") or {}).get("technical_details") or (debug.get("debug") or {}).get("trace", {}).get("technical_details")
            row = {
                "case": case_id,
                "status": status,
                "answer_preview": answer[:240],
                "technical_details": tech,
                "pending_interrupt_type": interrupt.get("type"),
            }
            results.append(row)
            print(json.dumps(row, indent=2))

        f_row = next((r for r in reversed(results) if r.get("technical_details")), results[0] if results else {})
        f_row = {**f_row, "case": "f"}
        results.append(f_row)
        print("=== summary ===")
        print(json.dumps(results, indent=2))
        Path("/opt/cursor/artifacts/live_auto_results.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())