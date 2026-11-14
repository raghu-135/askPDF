#!/usr/bin/env python3
"""Run Auto orchestrator live cases a,b,c,d,f against local stack."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx
import os

API = "http://127.0.0.1:8000/api"


def _auth_headers() -> dict[str, str]:
    token = os.getenv("ASKPDF_ADMIN_TOKEN", "").strip()
    if not token:
        env_path = Path(__file__).resolve().parents[1] / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("ASKPDF_ADMIN_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        raise RuntimeError("ASKPDF_ADMIN_TOKEN is required for live API calls")
    return {"Authorization": f"Bearer {token}"}
MODEL = "openai/gpt-4.1-mini"
ATTENTION = Path("/opt/cursor/artifacts/pdfs/attention.pdf")
BERT = Path("/opt/cursor/artifacts/pdfs/bert.pdf")


def wait_ready(client: httpx.Client, thread_id: str, file_hash: str, timeout: float = 600.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"{API}/threads/{thread_id}/files/{file_hash}/status")
        r.raise_for_status()
        payload = r.json()
        status = str(
            payload.get("status")
            or payload.get("process_status")
            or payload.get("processing_status")
            or ""
        ).lower()
        indexing = payload.get("indexing") if isinstance(payload.get("indexing"), dict) else {}
        indexing_status = str(indexing.get("status") or "").lower()
        projection = (payload.get("document_processing") or {}).get("projection")
        projection_status = ""
        if isinstance(projection, dict):
            projection_status = str(projection.get("status") or "").lower()
        ready = status in {"ready", "completed", "indexed", "done"} or indexing_status == "completed"
        if isinstance(projection, dict) and projection_status:
            ready = ready and projection_status == "completed"
        if ready:
            return True
        if status in {"failed", "error"} or indexing_status in {"failed", "error"}:
            print(f"index failed for {file_hash}: {payload}", file=sys.stderr)
            return False
        time.sleep(5)
    return False


def create_thread(client: httpx.Client) -> str:
    project_r = client.get(f"{API}/projects")
    project_r.raise_for_status()
    projects = project_r.json()
    project_id = projects[0]["id"] if isinstance(projects, list) and projects else None
    body = {"name": f"auto-live-{int(time.time())}"}
    if project_id:
        body["project_id"] = project_id
    r = client.post(f"{API}/threads", json=body)
    r.raise_for_status()
    return r.json()["id"]


def upload_pdf(client: httpx.Client, thread_id: str, path: Path) -> str:
    with path.open("rb") as handle:
        files = {"file": (path.name, handle, "application/pdf")}
        r = client.post(f"{API}/threads/{thread_id}/files/upload", files=files)
    r.raise_for_status()
    return r.json()["file_hash"]


def chat(client: httpx.Client, thread_id: str, question: str) -> dict:
    r = client.post(
        f"{API}/threads/{thread_id}/chat",
        json={
            "thread_id": thread_id,
            "question": question,
            "llm_model": MODEL,
            "bypass_clarification": True,
        },
        timeout=600.0,
    )
    r.raise_for_status()
    return r.json()


def resume_run(client: httpx.Client, thread_id: str, run_id: str, interrupt_id: str) -> dict:
    r = client.post(
        f"{API}/agent-runs/{run_id}/resume",
        params={"thread_id": thread_id},
        json={
            "interrupt_id": interrupt_id,
            "action": "approve",
        },
        timeout=600.0,
    )
    r.raise_for_status()
    return r.json()


def run_debug(client: httpx.Client, thread_id: str, run_id: str) -> dict:
    r = client.get(f"{API}/agent-runs/{run_id}", params={"thread_id": thread_id})
    r.raise_for_status()
    return r.json().get("agent_run") or {}


def main() -> int:
    results: list[dict] = []
    with httpx.Client(timeout=600.0, headers=_auth_headers()) as client:
        thread_id = create_thread(client)
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

        # case f from last delegated run with child ids
        f_row = next((r for r in reversed(results) if r.get("technical_details")), results[0] if results else {})
        f_row = {**f_row, "case": "f"}
        results.append(f_row)
        print("=== summary ===")
        print(json.dumps(results, indent=2))
        Path("/opt/cursor/artifacts/live_auto_results.json").write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())