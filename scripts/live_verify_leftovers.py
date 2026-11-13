#!/usr/bin/env python3
"""Live API verification for leftover fixes (OpenRouter stack)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("ASKPDF_E2E_ARTIFACTS", "/opt/cursor/artifacts"))
ART.mkdir(parents=True, exist_ok=True)
LOG = ART / "verification-leftovers-results.json"

API = os.environ.get("ASKPDF_E2E_API_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("ASKPDF_E2E_LLM_MODEL", "openai/gpt-4.1-nano")
PDF = Path(os.environ.get("ASKPDF_E2E_PDF", "/tmp/attention.pdf"))
DECLINE = "The attached documents do not address this question."


def load_admin() -> str:
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ASKPDF_ADMIN_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ASKPDF_ADMIN_TOKEN missing from .env")


ADMIN = load_admin()


def http_json(method: str, path: str, payload: dict | None = None, extra_headers: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {ADMIN}", "Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"{API}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode())


def upload_pdf(thread_id: str) -> str:
    cmd = [
        "curl",
        "-sf",
        "-X",
        "POST",
        f"{API}/api/threads/{thread_id}/files/upload",
        "-H",
        f"Authorization: Bearer {ADMIN}",
        "-F",
        f"file=@{PDF}",
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)["file_hash"]


def wait_ready(thread_id: str, file_hash: str, timeout_s: int = 600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            body = http_json("GET", f"/api/threads/{thread_id}/files/{file_hash}/status")
        except urllib.error.HTTPError:
            time.sleep(3)
            continue
        if body.get("status") == "ready":
            return
        time.sleep(3)
    raise TimeoutError("indexing did not reach ready")


def set_workflow(thread_id: str, workflow_id: str) -> None:
    http_json(
        "PUT",
        f"/api/threads/{thread_id}/settings",
        {"agent_workflow": {"workflow_id": workflow_id}},
    )


def chat(thread_id: str, question: str) -> tuple[int, str, dict]:
    started = time.perf_counter()
    body = http_json(
        "POST",
        f"/api/threads/{thread_id}/chat",
        {
            "thread_id": thread_id,
            "question": question,
            "llm_model": MODEL,
            "use_web_search": False,
            "use_reranker": True,
            "context_window": 64000,
        },
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    answer = str(body.get("answer") or body.get("final_answer") or "")
    if not answer:
        assistant = body.get("assistant_message") or {}
        answer = str(assistant.get("content") or "")
    return elapsed_ms, answer, body


def browser_capture_blank_status() -> dict:
    cmd = [
        "curl",
        "-s",
        "-o",
        "/tmp/capture-body.json",
        "-w",
        "%{http_code}",
        "-X",
        "POST",
        "http://127.0.0.1:8090/capture",
    ]
    code = subprocess.check_output(cmd, text=True).strip()
    detail = {}
    try:
        detail = json.loads(Path("/tmp/capture-body.json").read_text(encoding="utf-8"))
    except Exception:
        detail = {"raw": Path("/tmp/capture-body.json").read_text(encoding="utf-8")[:500]}
    return {"http_code": int(code), "detail": detail}


def main() -> int:
    results: dict = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()}
    project = http_json(
        "POST",
        "/api/projects",
        {"name": "leftovers-live", "description": "live verify", "embedding_model": "BAAI/bge-m3"},
    )
    thread = http_json("POST", f"/api/projects/{project['id']}/threads", {"name": "verify"})
    thread_id = thread["id"]
    file_hash = upload_pdf(thread_id)
    wait_ready(thread_id, file_hash)
    results["thread_id"] = thread_id
    results["file_hash"] = file_hash

    results["items"] = {}

    cap = browser_capture_blank_status()
    results["items"]["1_browser_capture_about_blank"] = {
        "pass": cap["http_code"] == 422,
        "http_code": cap["http_code"],
        "detail": cap["detail"],
        "timing_ms": 0,
    }

    workflow_rows = {}
    for workflow in ("router_rag_agent", "orchestrator_worker_rag_agent"):
        set_workflow(thread_id, workflow)
        oos_ms, oos_answer, _ = chat(thread_id, "What is the population of Tokyo?")
        in_ms, in_answer, _ = chat(
            thread_id,
            "For the Transformer (base model) in Table 2, what is the EN-FR training cost in FLOPs? Reply with only the numeric value in scientific notation.",
        )
        workflow_rows[workflow] = {
            "out_of_scope_ms": oos_ms,
            "out_of_scope_answer": oos_answer,
            "out_of_scope_pass": DECLINE in oos_answer,
            "in_scope_ms": in_ms,
            "in_scope_answer": in_answer,
            "in_scope_pass": "3.3" in in_answer and "10^18" in in_answer.replace(" ", "").replace("×", "x"),
        }

    results["items"]["2_scope_decline"] = {
        "pass": all(row["out_of_scope_pass"] for row in workflow_rows.values()),
        "workflows": workflow_rows,
    }
    results["items"]["3_table2_enfr"] = {
        "pass": all(row["in_scope_pass"] for row in workflow_rows.values()),
        "workflows": workflow_rows,
    }

    hermes = http_json(
        "POST",
        f"/api/threads/{thread_id}/agent-tasks",
        {
            "definition_id": "hermes_rag_agent",
            "objective": "Reply with exactly: cost-check",
            "llm_model": MODEL,
            "context_window": 32000,
            "web_search_mode": "off",
        },
        extra_headers={"Idempotency-Key": f"cost-{int(time.time())}"},
    )
    task_id = hermes["task"]["id"]
    version = hermes["task"]["version"]
    http_json(
        "POST",
        f"/api/agent-tasks/{task_id}/commands/start?thread_id={thread_id}",
        {"expected_version": version},
        extra_headers={"Idempotency-Key": f"cost-start-{int(time.time())}"},
    )
    cost = None
    for _ in range(120):
        task_body = http_json("GET", f"/api/agent-tasks/{task_id}?thread_id={thread_id}")
        status = task_body["task"]["status"]
        if status in {"completed", "failed", "cancelled"}:
            metrics = ((task_body["task"].get("active_run") or {}).get("trace") or {}).get("metrics") or {}
            cost = metrics.get("llm_provider_cost_usd")
            break
        time.sleep(2)
    results["items"]["4_openrouter_cost"] = {
        "pass": cost is not None and float(cost) > 0,
        "llm_provider_cost_usd": cost,
    }

    LOG.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
