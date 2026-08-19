"""Short-lived signed execution context for MCP clients without metadata support."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import time
from typing import Any, Mapping

from app.tools.context import ToolInvocationContext


TOKEN_ARGUMENT = "_askpdf_context_token"


def _secret() -> bytes:
    value = os.getenv("HERMES_MCP_CONTEXT_SECRET", "").encode()
    if len(value) < 32:
        raise ValueError("HERMES_MCP_CONTEXT_SECRET must contain at least 32 characters")
    return value


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def issue_execution_context_token(
    context: ToolInvocationContext,
    *,
    task_id: str,
    allowed_tools: list[str],
    ttl_seconds: int = 3600,
) -> str:
    payload = {
        "v": 1,
        "exp": int(time.time()) + max(60, ttl_seconds),
        "task_id": task_id,
        "allowed_tools": sorted(set(allowed_tools)),
        "context": context.as_dict(),
    }
    encoded = _encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    signature = _encode(hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest())
    return f"{encoded}.{signature}"


def decode_execution_context_token(token: str, *, tool_name: str) -> ToolInvocationContext:
    try:
        encoded, signature = token.split(".", 1)
        expected = _encode(hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature mismatch")
        payload: Mapping[str, Any] = json.loads(_decode(encoded))
        if int(payload.get("v") or 0) != 1 or int(payload.get("exp") or 0) < int(time.time()):
            raise ValueError("token expired")
        if tool_name not in set(str(value) for value in payload.get("allowed_tools") or []):
            raise ValueError("tool is not allowed for this execution")
        context = payload.get("context")
        if not payload.get("task_id") or not isinstance(context, Mapping):
            raise ValueError("token context is incomplete")
        return ToolInvocationContext.from_mapping(context)
    except (TypeError, ValueError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError) as exc:
        raise ValueError("Invalid Hermes MCP execution context") from exc
