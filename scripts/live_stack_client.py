"""Shared HTTP helpers for live-stack scripts (e2e, benchmarks, proofs)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Optional

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API = os.getenv("ASKPDF_API_URL", "http://127.0.0.1:8000/api").rstrip("/")


def admin_token() -> str:
    token = os.getenv("ASKPDF_ADMIN_TOKEN", "").strip()
    if token:
        return token
    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ASKPDF_ADMIN_TOKEN="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if not token:
        raise RuntimeError("ASKPDF_ADMIN_TOKEN is required for live API calls")
    return token


def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {admin_token()}"}


def project_id(projects_payload: Any) -> Optional[str]:
    rows = (
        projects_payload
        if isinstance(projects_payload, list)
        else (projects_payload.get("projects") if isinstance(projects_payload, dict) else None)
    )
    if isinstance(rows, list) and rows:
        return str(rows[0].get("id") or "") or None
    return None


def wait_indexed(
    client: httpx.Client,
    *,
    thread_id: str,
    file_hash: str,
    timeout: float = 1200.0,
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.get(f"/threads/{thread_id}/files/{file_hash}/status").json()
        status = str(
            payload.get("status")
            or payload.get("process_status")
            or payload.get("processing_status")
            or ""
        ).lower()
        indexing = payload.get("indexing") if isinstance(payload.get("indexing"), dict) else {}
        indexing_status = str(indexing.get("status") or "").lower()
        projection = (payload.get("document_processing") or {}).get("projection")
        projection_status = str((projection or {}).get("status") or "").lower() if isinstance(projection, dict) else ""
        ready = status in {"ready", "completed", "indexed", "done"} or indexing_status == "completed"
        if isinstance(projection, dict) and projection_status:
            ready = ready and projection_status == "completed"
        if ready:
            return
        if status in {"failed", "error"} or indexing_status in {"failed", "error"}:
            raise RuntimeError(f"index failed for {file_hash}: {payload}")
        time.sleep(5)
    raise TimeoutError(f"indexing timeout for {file_hash}")


def update_thread_settings(client: httpx.Client, thread_id: str, updates: dict[str, Any]) -> None:
    """Merge updates onto persisted thread chat settings."""
    if not updates:
        return
    client.put(
        f"/threads/{thread_id}/settings",
        json=updates,
    ).raise_for_status()


def create_thread(
    client: httpx.Client,
    name: str,
    *,
    workflow_id: Optional[str] = None,
    settings_patch: Optional[dict[str, Any]] = None,
) -> str:
    pid = project_id(client.get("/projects").json())
    body: dict[str, Any] = {"name": name}
    if pid:
        body["project_id"] = pid
    thread_id = client.post("/threads", json=body).json()["id"]
    merged_patch: dict[str, Any] = dict(settings_patch or {})
    if workflow_id:
        merged_patch = {**merged_patch, "agent_workflow": {"workflow_id": workflow_id}}
    update_thread_settings(client, thread_id, merged_patch)
    return thread_id


def upload_pdf(client: httpx.Client, thread_id: str, path: Path) -> str:
    with path.open("rb") as handle:
        client.post(
            f"/threads/{thread_id}/files/upload",
            files={"file": (path.name, handle, "application/pdf")},
        ).raise_for_status()
    listed = client.get(f"/threads/{thread_id}/files").json()
    for row in listed.get("files") or []:
        if row.get("file_name") == path.name:
            return str(row["file_hash"])
    return str(listed["files"][-1]["file_hash"])


def upload_and_wait(client: httpx.Client, thread_id: str, path: Path) -> str:
    file_hash = upload_pdf(client, thread_id, path)
    wait_indexed(client, thread_id=thread_id, file_hash=file_hash)
    return file_hash


def thread_chat(
    client: httpx.Client,
    thread_id: str,
    question: str,
    *,
    llm_model: str,
    bypass_clarification: bool = True,
    chat_options: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "thread_id": thread_id,
        "question": question,
        "llm_model": llm_model,
        "bypass_clarification": bypass_clarification,
    }
    if chat_options:
        payload.update(chat_options)
    response = client.post(
        f"/threads/{thread_id}/chat",
        json=payload,
        timeout=900.0,
    )
    response.raise_for_status()
    body = response.json()
    run_id = body.get("agent_run_id")
    interrupt = body.get("pending_interrupt") or {}
    if body.get("status") == "awaiting_human" and interrupt.get("interrupt_id") and run_id:
        client.post(
            f"/agent-runs/{run_id}/resume",
            params={"thread_id": thread_id},
            json={"interrupt_id": interrupt["interrupt_id"], "action": "approve"},
            timeout=900.0,
        ).raise_for_status()
    return body


def get_agent_run(client: httpx.Client, thread_id: str, run_id: str) -> dict[str, Any]:
    return client.get(f"/agent-runs/{run_id}", params={"thread_id": thread_id}).json().get("agent_run") or {}


def scored_agent_run(client: httpx.Client, thread_id: str, run_id: str) -> dict[str, Any]:
    """Prefer child router/delegate run when Auto orchestrator delegates."""
    detail = get_agent_run(client, thread_id, run_id)
    child_id = (detail.get("debug") or {}).get("technical_details", {}).get("child_run_id")
    if child_id:
        return get_agent_run(client, thread_id, str(child_id))
    return detail


def answer_from_run(detail: dict[str, Any]) -> str:
    final = detail.get("final_output") or {}
    if isinstance(final, dict) and final.get("answer"):
        return str(final["answer"])
    turns = detail.get("turns") or []
    if turns:
        return str(turns[-1].get("answer") or "")
    return ""


def pages_from_document_sources(sources: Any) -> set[int]:
    """Expand page numbers from chat `document_sources` (matches trace refs_from_documents)."""
    pages: set[int] = set()
    if not isinstance(sources, list):
        return pages
    for item in sources:
        if not isinstance(item, dict):
            continue
        for key in ("page_number", "page", "page_start", "page_end"):
            value = item.get(key)
            if value not in (None, ""):
                try:
                    pages.add(int(value))
                except (TypeError, ValueError):
                    pass
        raw_pages = item.get("pages")
        if isinstance(raw_pages, list):
            for page in raw_pages:
                try:
                    pages.add(int(page))
                except (TypeError, ValueError):
                    pass
        elif isinstance(raw_pages, str) and raw_pages.strip():
            for part in raw_pages.replace("–", "-").split(","):
                part = part.strip()
                if not part:
                    continue
                if "-" in part:
                    start, end = part.split("-", 1)
                    try:
                        pages.update(range(int(start), int(end) + 1))
                    except ValueError:
                        continue
                else:
                    try:
                        pages.add(int(part))
                    except ValueError:
                        pass
    return pages


def live_client(api_base: Optional[str] = None, **kwargs: Any) -> httpx.Client:
    return httpx.Client(
        base_url=(api_base or DEFAULT_API),
        headers=auth_headers(),
        timeout=kwargs.pop("timeout", 900.0),
        **kwargs,
    )