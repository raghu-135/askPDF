"""Minimal single-user authentication for the self-hosted control plane."""

from __future__ import annotations

import hmac
import os
from contextvars import ContextVar

from fastapi import Request


_principal: ContextVar[str | None] = ContextVar("askpdf_principal", default=None)


def _bearer(value: str | None) -> str | None:
    if not value or not value.startswith("Bearer "):
        return None
    token = value[7:]
    if not token or token != token.strip() or any(char.isspace() for char in token):
        return None
    return token


def authenticate(request: Request) -> str | None:
    """Authenticate a request using the configured admin token or trusted proxy subject."""
    if os.getenv("ASKPDF_TRUST_PROXY_AUTH", "false").lower() in {"1", "true", "yes", "on"}:
        subject = (request.headers.get("x-authenticated-user") or "").strip()
        if subject:
            return subject[:255]
    supplied = _bearer(request.headers.get("authorization"))
    expected = os.getenv("ASKPDF_ADMIN_TOKEN", "").strip()
    if supplied and expected and hmac.compare_digest(supplied, expected):
        return "admin"
    return None


def current_principal() -> str | None:
    return _principal.get()


def set_principal(value: str | None):
    return _principal.set(value)


def reset_principal(token) -> None:
    _principal.reset(token)
