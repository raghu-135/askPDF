#!/usr/bin/env python3
"""Live verification matrix for Attention PDF (router + orchestrator)."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("ASKPDF_E2E_ARTIFACTS", "/opt/cursor/artifacts"))
ART.mkdir(parents=True, exist_ok=True)
API = os.environ.get("ASKPDF_E2E_API_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("ASKPDF_E2E_LLM_MODEL", "openai/gpt-4.1-mini")
PDF = Path(os.environ.get("ASKPDF_E2E_PDF", "/tmp/attention.pdf"))
DECLINE = "The attached documents do not address this question."


def load_admin() -> str:
    token = os.environ.get("ASKPDF_ADMIN_TOKEN", "").strip()
    if token:
        return token
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        if line.startswith("ASKPDF_ADMIN_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("ASKPDF_ADMIN_TOKEN missing")


ADMIN = load_admin()


def http_json(method: str, path: str, payload: dict | None = None) -> dict:
    headers = {"Authorization": f"Bearer {ADMIN}", "Content-Type": "application/json"}
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"{API}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read().decode())


def indexing_ready(status_body: dict) -> bool:
    parsing = str((status_body.get("parsing") or {}).get("status") or "")
    indexing = str((status_body.get("indexing") or {}).get("status") or "")
    if parsing in {"completed", "ready"} and indexing in {"completed", "ready"}:
        return True
    projection = (status_body.get("document_processing") or {}).get("projection") or {}
    vectors = str(projection.get("vector_status") or "")
    if str(projection.get("status") or "") == "completed" and vectors in {"ready", "indexed"}:
        return True
    return str(status_body.get("status") or "") == "ready"


def upload_and_index(thread_id: str) -> str:
    out = subprocess.check_output(
        [
            "curl", "-sf", "-X", "POST",
            f"{API}/api/threads/{thread_id}/files/upload",
            "-H", f"Authorization: Bearer {ADMIN}",
            "-F", f"file=@{PDF}",
        ],
        text=True,
    )
    file_hash = json.loads(out)["file_hash"]
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            st = http_json("GET", f"/api/threads/{thread_id}/files/{file_hash}/status")
        except urllib.error.HTTPError:
            time.sleep(4)
            continue
        if indexing_ready(st):
            return file_hash
        fail = str(st.get("status") or "")
        parsing = str((st.get("parsing") or {}).get("status") or "")
        indexing = str((st.get("indexing") or {}).get("status") or "")
        if fail in {"failed", "error"} or parsing in {"failed", "error"} or indexing in {"failed", "error"}:
            http_json("POST", f"/api/threads/{thread_id}/files/{file_hash}/retry")
        time.sleep(4)
    raise TimeoutError("indexing timeout")


def set_workflow(thread_id: str, workflow_id: str) -> None:
    http_json("PUT", f"/api/threads/{thread_id}/settings", {"agent_workflow": {"workflow_id": workflow_id}})


def chat(thread_id: str, question: str) -> tuple[int, str, dict]:
    t0 = time.perf_counter()
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
    ms = int((time.perf_counter() - t0) * 1000)
    answer = str(body.get("answer") or body.get("final_answer") or "")
    if not answer and isinstance(body.get("assistant_message"), dict):
        answer = str(body["assistant_message"].get("content") or "")
    run_id = str(body.get("run_id") or body.get("agent_run_id") or "")
    if not answer.strip() and run_id:
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                payload = http_json("GET", f"/api/agent-runs/{run_id}?thread_id={thread_id}")
            except urllib.error.HTTPError:
                time.sleep(1)
                continue
            agent_run = payload.get("agent_run") if isinstance(payload.get("agent_run"), dict) else payload
            if isinstance(agent_run, dict):
                answer = str(
                    agent_run.get("final_answer")
                    or (agent_run.get("result") if isinstance(agent_run.get("result"), dict) else {}).get("final_answer")
                    or answer
                )
                status = str(agent_run.get("status") or "").casefold()
                if answer.strip() or status in {"completed", "failed", "cancelled"}:
                    break
            time.sleep(2)
    if not answer.strip():
        try:
            msgs = http_json("GET", f"/api/threads/{thread_id}/messages?limit=5")
            for item in reversed(msgs if isinstance(msgs, list) else msgs.get("messages") or []):
                if isinstance(item, dict) and str(item.get("role") or "") == "assistant":
                    answer = str(item.get("content") or answer)
                    if answer.strip():
                        break
        except urllib.error.HTTPError:
            pass
    return ms, answer, body


def _write_screenshot(
    path: Path,
    workflow: str,
    scenario: str,
    question: str,
    answer: str,
    passed: bool,
) -> None:
    lines = [
        f"workflow: {workflow}",
        f"scenario: {scenario}",
        f"pass: {passed}",
        "",
        f"Q: {question[:200]}",
        "",
        f"A: {answer[:1200]}",
    ]
    text = "\n".join(lines)
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("RGB", (900, 420), color=(248, 249, 251))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        y = 12
        for line in text.splitlines():
            draw.text((12, y), line[:110], fill=(20, 20, 24), font=font)
            y += 16
        img.save(path)
    except Exception:
        path.with_suffix(".txt").write_text(text, encoding="utf-8")


def export_trace_scores(thread_id: str, run_id: str) -> list[dict]:
    try:
        payload = http_json("GET", f"/api/agent-runs/{run_id}?thread_id={thread_id}")
    except urllib.error.HTTPError:
        return []
    agent_run = payload.get("agent_run") if isinstance(payload.get("agent_run"), dict) else payload
    debug = agent_run.get("debug") if isinstance(agent_run, dict) else {}
    trace = debug.get("trace") if isinstance(debug, dict) else {}
    ops = trace.get("operations") if isinstance(trace.get("operations"), list) else []
    scores: list[dict] = []
    for op in ops:
        output = op.get("output") if isinstance(op.get("output"), dict) else {}
        if output.get("evidence_packet_scores"):
            scores.extend(output["evidence_packet_scores"])
    return scores


EXTRA_SCENARIOS_BY_WORKFLOW = {
    "router_rag_agent": [
        (
            "architecture",
            "What architecture does the paper propose? Reply with the architecture name only.",
            lambda a: (
                DECLINE not in a
                and "does not specify" not in a.casefold()
                and re.search(r"\btransformer\b", a or "", re.I) is not None
            ),
        ),
    ],
    "corrective_self_rag_agent": [
        (
            "paper_title",
            "What is the title of this paper? Reply with the full title only.",
            lambda a: (
                DECLINE not in a
                and "could not verify" not in a.casefold()
                and "Unresolved evidence gaps" not in a
                and "attention is all you need" in (a or "").casefold()
            ),
        ),
    ],
}


SCENARIOS = [
    ("tokyo", "What is the population of Tokyo?", lambda a: a.strip() == DECLINE),
    (
        "encoder",
        "How many encoder layers does the base Transformer use? Reply with the number only.",
        lambda a: DECLINE not in a and re.search(r"\b6\b", a or "") is not None,
    ),
    (
        "table2_big_enfr",
        "For the Transformer (big model) in Table 2, what is the EN-FR training cost in FLOPs? Reply with the value only.",
        lambda a: DECLINE not in a and "2.3" in a and "10^19" in a.replace(" ", "").replace("×", "x"),
    ),
    (
        "table2_base_enfr",
        "For the Transformer (base model) in Table 2, what is the EN-FR training cost in FLOPs? Reply with the value only.",
        lambda a: DECLINE not in a and "3.3" in a and "10^18" in a.replace(" ", "").replace("×", "x"),
    ),
    (
        "bleu_en_de",
        "What is the EN-DE BLEU score for the big Transformer model in Table 2? Reply with 28.4 only if that is what the paper states.",
        lambda a: DECLINE not in a and "28.4" in a and len(a.strip()) > 0,
    ),
]


def main() -> int:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    project = http_json("POST", "/api/projects", {"name": "live-matrix", "description": "attention", "embedding_model": "BAAI/bge-m3"})
    thread = http_json("POST", f"/api/projects/{project['id']}/threads", {"name": "matrix"})
    thread_id = thread["id"]
    upload_and_index(thread_id)

    rows: list[dict] = []
    workflow_order = (
        "router_rag_agent",
        "orchestrator_worker_rag_agent",
        "corrective_self_rag_agent",
    )
    for workflow in workflow_order:
        set_workflow(thread_id, workflow)
        scenario_list = [
            *SCENARIOS,
            *EXTRA_SCENARIOS_BY_WORKFLOW.get(workflow, []),
        ]
        for key, question, pred in scenario_list:
            print(f"RUN {workflow}/{key} ...", flush=True)
            ms, answer, body = chat(thread_id, question)
            print(f"DONE {workflow}/{key} pass={bool(pred(answer))} ms={ms}", flush=True)
            run_id = str(body.get("run_id") or body.get("agent_run_id") or "")
            scores = export_trace_scores(thread_id, run_id) if run_id else []
            passed = bool(pred(answer))
            row = {
                "workflow": workflow,
                "scenario": key,
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
            shot = ART / f"live-{workflow}-{key}.json"
            shot.write_text(json.dumps({"answer": answer, "row": row}, indent=2), encoding="utf-8")
            _write_screenshot(ART / f"live-{workflow}-{key}.png", workflow, key, question, answer, passed)

    out = {"sha": sha, "model": MODEL, "thread_id": thread_id, "rows": rows}
    (ART / "live-attention-matrix.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())