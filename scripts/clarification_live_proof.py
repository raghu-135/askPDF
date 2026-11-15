#!/usr/bin/env python3
"""Exercise clarification card flow via chat API for live proof."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
MODEL = "inclusionai/ling-3.2-flash:free"
API = "http://127.0.0.1:8000/api"


def _token() -> str:
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ASKPDF_ADMIN_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"')
    raise RuntimeError("ASKPDF_ADMIN_TOKEN missing")


def main() -> int:
    headers = {"Authorization": f"Bearer {_token()}"}
    out_path = Path("/opt/cursor/artifacts/clarification_live_proof.json")
    with httpx.Client(timeout=180.0, headers=headers) as client:
        projects = client.get(f"{API}/projects").json()
        project_id = projects[0]["id"] if isinstance(projects, list) and projects else None
        body = {"name": f"clarify-proof-{int(time.time())}"}
        if project_id:
            body["project_id"] = project_id
        thread_id = client.post(f"{API}/threads", json=body).json()["id"]
        pdf_path = REPO / "rag_service/tests/01030000000000.pdf"
        with pdf_path.open("rb") as handle:
            client.post(
                f"{API}/threads/{thread_id}/files/upload",
                files={"file": ("sample.pdf", handle, "application/pdf")},
            ).raise_for_status()
        for _ in range(120):
            status = client.get(f"{API}/threads/{thread_id}/indexing/status").json()
            if status.get("status") == "ready":
                break
            time.sleep(2)
        ambiguous = "Compare them."
        first = client.post(
            f"{API}/threads/{thread_id}/chat",
            json={
                "thread_id": thread_id,
                "question": ambiguous,
                "llm_model": MODEL,
                "agent_workflow_id": "router_rag_agent",
            },
        ).json()
        proof = {
            "thread_id": thread_id,
            "model": MODEL,
            "ambiguous_question": ambiguous,
            "first": {
                "status": first.get("status"),
                "clarification_options": first.get("clarification_options"),
                "answer": first.get("answer"),
                "agent_run_id": first.get("agent_run_id"),
            },
        }
        options = list(first.get("clarification_options") or [])
        if len(options) < 2:
            proof["error"] = "expected clarification_options from ambiguous Auto question"
            out_path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
            print(json.dumps(proof, indent=2))
            return 1
        picked = options[0]
        second = client.post(
            f"{API}/threads/{thread_id}/chat",
            json={
                "thread_id": thread_id,
                "question": picked,
                "llm_model": MODEL,
                "agent_workflow_id": "auto_rag_agent",
                "bypass_clarification": True,
            },
        ).json()
        proof["picked_option"] = picked
        proof["second"] = {
            "status": second.get("status"),
            "answer_preview": (second.get("answer") or "")[:400],
            "agent_run_id": second.get("agent_run_id"),
        }
        custom = "List every document file currently attached to this thread."
        third = client.post(
            f"{API}/threads/{thread_id}/chat",
            json={
                "thread_id": thread_id,
                "question": custom,
                "llm_model": MODEL,
                "agent_workflow_id": "auto_rag_agent",
                "bypass_clarification": True,
            },
        ).json()
        proof["custom_question"] = custom
        proof["third"] = {
            "status": third.get("status"),
            "answer_preview": (third.get("answer") or "")[:400],
        }
    out_path.write_text(json.dumps(proof, indent=2), encoding="utf-8")
    print(json.dumps(proof, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())