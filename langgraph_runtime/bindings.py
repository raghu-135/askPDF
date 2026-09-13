"""Signed opaque continuation bindings owned by langgraph-runtime."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

from runtime_protocol.errors import RuntimeError


def _secret() -> bytes:
    value = os.getenv("LANGGRAPH_RUNTIME_BINDING_SECRET", "").encode()
    if len(value) < 32:
        raise RuntimeError(
            "runtime_configuration_invalid",
            "LANGGRAPH_RUNTIME_BINDING_SECRET must contain at least 32 characters",
        )
    return value


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_binding(*, checkpoint_thread_id: str, run_id: str, ttl_seconds: int = 2_592_000) -> str:
    payload = _encode(json.dumps({
        "v": 1,
        "run_id": run_id,
        "checkpoint_ref": checkpoint_thread_id,
        "exp": int(time.time()) + max(300, ttl_seconds),
    }, sort_keys=True, separators=(",", ":")).encode())
    signature = _encode(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
    return f"lgb1.{payload}.{signature}"


def resolve_binding(binding_id: str, *, run_id: str | None = None) -> str:
    try:
        prefix, payload, signature = binding_id.split(".", 2)
        if prefix != "lgb1":
            raise ValueError
        expected = _encode(hmac.new(_secret(), payload.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError
        value: dict[str, Any] = json.loads(_decode(payload))
        if int(value.get("exp") or 0) < int(time.time()):
            raise RuntimeError("runtime_binding_expired", "Runtime continuation has expired")
        if run_id is not None and str(value.get("run_id") or "") != run_id:
            raise ValueError
        checkpoint_ref = str(value.get("checkpoint_ref") or "")
        if not checkpoint_ref:
            raise ValueError
        return checkpoint_ref
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("runtime_binding_invalid", "Runtime continuation binding is invalid") from exc
