"""Shared constants and helpers for live HTTP/matrix verification scripts."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = Path(os.environ.get("ASKPDF_E2E_ARTIFACTS", "/opt/cursor/artifacts"))
API = os.environ.get("ASKPDF_E2E_API_URL", "http://127.0.0.1:8000")
APP = os.environ.get("ASKPDF_E2E_APP_URL", "http://127.0.0.1:3000")
DEFAULT_PDF = Path(os.environ.get("ASKPDF_E2E_PDF", "/tmp/attention.pdf"))


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