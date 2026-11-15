#!/usr/bin/env python3
"""One thread per live Auto case; collects answers and child_run_id for proof."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_stack_client import (  # noqa: E402
    answer_from_run,
    create_thread,
    get_agent_run,
    live_client,
    thread_chat,
    upload_and_wait,
)

REPO = Path(__file__).resolve().parents[1]
ATTENTION = Path("/opt/cursor/artifacts/pdfs/attention.pdf")
BERT = Path("/opt/cursor/artifacts/pdfs/bert.pdf")
MODEL = "openai/gpt-4.1-mini"
DECLINE = "The attached documents do not address this question."


def _setup_thread(client: httpx.Client, label: str) -> str:
    thread_id = create_thread(client, label)
    for path in (ATTENTION, BERT):
        upload_and_wait(client, thread_id, path)
    return thread_id


def _chat(client: httpx.Client, thread_id: str, question: str, *, approve: bool = False) -> dict:
    if approve:
        out = client.post(
            f"/threads/{thread_id}/chat",
            json={
                "thread_id": thread_id,
                "question": question,
                "llm_model": MODEL,
                "bypass_clarification": True,
            },
            timeout=600.0,
        ).json()
        run_id = out.get("agent_run_id")
        interrupt = out.get("pending_interrupt") or {}
        if out.get("status") == "awaiting_human" and interrupt.get("interrupt_id") and run_id:
            client.post(
                f"/agent-runs/{run_id}/resume",
                json={"thread_id": thread_id, "interrupt_id": interrupt["interrupt_id"], "action": "approve"},
            ).raise_for_status()
    else:
        out = thread_chat(client, thread_id, question, llm_model=MODEL)
        run_id = out.get("agent_run_id")
    detail = get_agent_run(client, thread_id, str(run_id)) if run_id else {}
    answer = answer_from_run(detail)
    tech = (detail.get("debug") or {}).get("technical_details") or {}
    ops = [row.get("operation_id") for row in (detail.get("debug") or {}).get("operations") or []]
    return {
        "thread_id": thread_id,
        "run_id": run_id,
        "status": detail.get("status"),
        "answer": answer,
        "technical_details": tech,
        "operations": ops,
        "interrupt_type": (out.get("pending_interrupt") or {}).get("type"),
        "child_run_id": tech.get("child_run_id"),
        "child_workflow_id": tech.get("child_workflow_id"),
    }


def main() -> int:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    stamp = int(time.time())
    cases = [
        (
            "simple-2pdf",
            f"e2e-simple-2pdf-{stamp}",
            "What is the scaled dot-product attention mechanism in the Attention paper?",
            False,
            lambda row: row["status"] == "completed"
            and row["interrupt_type"] is None
            and "plan_approval_gate" not in row["operations"]
            and row.get("child_workflow_id") == "router_rag_agent"
            and bool(row.get("child_run_id")),
        ),
        (
            "a",
            f"e2e-case-a-{stamp}",
            "What is the scaled dot-product attention mechanism described in the Attention paper?",
            False,
            lambda row: row["status"] == "completed"
            and "technical error" not in row["answer"].lower()
            and row.get("child_workflow_id") == "router_rag_agent"
            and bool(row.get("child_run_id")),
        ),
        (
            "b",
            f"e2e-case-b-{stamp}",
            "Compare the Attention paper and the BERT paper on architecture and training objectives.",
            False,
            lambda row: row["status"] == "completed"
            and row["interrupt_type"] is None
            and bool(row.get("child_run_id"))
            and row["child_workflow_id"] in {"orchestrator_worker_rag_agent", "router_rag_agent"},
        ),
        (
            "c",
            f"e2e-case-c-{stamp}",
            "Under Basel IV, what is the required minimum leverage ratio for a global systemically important bank in 2030?",
            False,
            lambda row: DECLINE in row["answer"]
            and row["status"] == "completed"
            and not row.get("child_run_id"),
        ),
        (
            "d",
            f"e2e-case-d-{stamp}",
            "Write a literature review comparing transformer attention with BERT's bidirectional encoding.",
            True,
            lambda row: row["interrupt_type"] == "plan_approval"
            and ("Deep Research task" in row["answer"] or "task `" in row["answer"]),
        ),
    ]
    rows: list[dict] = []
    child_ids: set[str] = set()
    with live_client() as client:
        for case_id, label, question, approve, predicate in cases:
            thread_id = _setup_thread(client, label)
            row = _chat(client, thread_id, question, approve=approve)
            cid = row.get("child_run_id")
            if cid:
                assert cid not in child_ids, f"duplicate child_run_id {cid} for case {case_id}"
                child_ids.add(cid)
            rows.append(
                {
                    "case": case_id,
                    "pass": predicate(row),
                    "thread_id": thread_id,
                    "thread_name": label,
                    "question": question,
                    "tested_sha": sha,
                    **row,
                }
            )
            print(json.dumps(rows[-1], indent=2)[:2000], file=sys.stderr)

    delegate_cases = [r for r in rows if r.get("child_run_id")]
    rows.append(
        {
            "case": "f",
            "pass": len(child_ids) >= 3 and len(child_ids) == len(delegate_cases),
            "thread_id": delegate_cases[-1].get("thread_id") if delegate_cases else None,
            "run_id": delegate_cases[-1].get("run_id") if delegate_cases else None,
            "technical_details": delegate_cases[-1].get("technical_details") if delegate_cases else None,
            "child_run_id": delegate_cases[-1].get("child_run_id") if delegate_cases else None,
            "distinct_child_run_ids": sorted(child_ids),
            "tested_sha": sha,
            "answer": "",
        }
    )
    out_path = Path("/opt/cursor/artifacts/live_auto_proof.json")
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0 if all(r.get("pass") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())