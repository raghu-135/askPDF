"""Shared constants and helpers for live HTTP/matrix verification scripts."""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("ASKPDF_E2E_ARTIFACTS", "/opt/cursor/artifacts"))
API = os.environ.get("ASKPDF_E2E_API_URL", "http://127.0.0.1:8000")
APP = os.environ.get("ASKPDF_E2E_APP_URL", "http://127.0.0.1:3000")
DEFAULT_PDF = Path(os.environ.get("ASKPDF_E2E_PDF", "/tmp/attention.pdf"))
# Pinned free model for UI/smoke (fixed context; must support tool calls). Not for quality matrices.
DEFAULT_FREE_LLM_MODEL = os.environ.get(
    "ASKPDF_E2E_FREE_LLM_MODEL",
    "inclusionai/ling-3.2-flash:free",
)

# OpenRouter :free tier is ~20 RPM account-wide; pace chat/matrix HTTP to reduce 429s.
_MIN_REQUEST_INTERVAL_S = float(os.environ.get("ASKPDF_E2E_MIN_REQUEST_INTERVAL_S", "3.2"))
_pace_lock = threading.Lock()
_last_request_monotonic = 0.0


def pace_live_llm_requests() -> None:
    """Sleep so successive live matrix/UI calls stay under account RPM limits."""

    global _last_request_monotonic
    with _pace_lock:
        now = time.monotonic()
        wait_s = _MIN_REQUEST_INTERVAL_S - (now - _last_request_monotonic)
        if wait_s > 0:
            time.sleep(wait_s)
        _last_request_monotonic = time.monotonic()


def http_json(
    method: str,
    url_prefix: str,
    path: str,
    payload: dict | None = None,
    *,
    admin_token: str,
    timeout_s: float = 1800.0,
    max_429_retries: int = 12,
) -> dict:
    headers = {"Authorization": f"Bearer {admin_token}", "Content-Type": "application/json"}
    data = None if payload is None else json.dumps(payload).encode()
    url = f"{url_prefix}{path}"
    for attempt in range(max_429_retries + 1):
        pace_live_llm_requests()
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < max_429_retries:
                retry_after = exc.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = max(float(retry_after), _MIN_REQUEST_INTERVAL_S)
                else:
                    delay = min(_MIN_REQUEST_INTERVAL_S * (2**attempt), 120.0)
                time.sleep(delay)
                continue
            raise
    raise RuntimeError(f"http_json exhausted 429 retries: {method} {path}")


def load_admin_token() -> str:
    token = os.environ.get("ASKPDF_ADMIN_TOKEN", "").strip()
    if token:
        return token
    env_path = REPO / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ASKPDF_ADMIN_TOKEN="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("ASKPDF_ADMIN_TOKEN missing")


def wait_langgraph_ready(
    *,
    url: str | None = None,
    timeout_s: float = 120.0,
    poll_s: float = 2.0,
) -> None:
    """Block until the LangGraph runtime reports ready (avoids flaky chat right after restarts)."""

    ready_url = url or os.environ.get("ASKPDF_E2E_LANGGRAPH_READY_URL", "http://127.0.0.1:8100/readyz")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(ready_url, timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(poll_s)
    raise TimeoutError(f"langgraph runtime not ready: {ready_url}")