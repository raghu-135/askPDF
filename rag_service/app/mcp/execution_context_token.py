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
from app.runtime.hermes_config import HermesConfigurationError, hermes_model_context_length


TOKEN_ARGUMENT = "_askpdf_context_token"
TOKEN_HEADER = "x-askpdf-execution-context"


class ExecutionContextTokenError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__("Invalid Hermes MCP execution context")
        self.reason = reason


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
    context_data = context.as_dict()
    extensions = dict(context_data.get("extensions") or {})
    extensions["task_id"] = task_id
    context_data["extensions"] = extensions
    payload = {
        "v": 1,
        "exp": int(time.time()) + max(60, ttl_seconds),
        "task_id": task_id,
        "allowed_tools": sorted(set(allowed_tools)),
        "model_settings": {
            "llm_model": extensions.get("llm_model"),
            "embedding_model": context.embedding_model,
            "context_window": context.context_window,
        },
        "context": context_data,
    }
    encoded = _encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    signature = _encode(hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest())
    return f"{encoded}.{signature}"


def validate_execution_context_identity(
    context: ToolInvocationContext,
    *,
    run_id: str,
    thread_id: str,
    task_id: str,
) -> None:
    """Bind a valid token to the exact runtime request admitting it."""
    extensions = dict(context.extensions or {})
    if any((
        str(context.run_id or "") != str(run_id),
        str(context.thread_id or "") != str(thread_id),
        str(extensions.get("task_id") or "") != str(task_id),
    )):
        raise ExecutionContextTokenError("identity_mismatch")


def decode_execution_context_token(token: str, *, tool_name: str | None = None) -> ToolInvocationContext:
    try:
        parts = token.split(".", 1)
        if len(parts) != 2 or not all(parts):
            raise ExecutionContextTokenError("malformed")
        encoded, signature = parts
        expected = _encode(hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ExecutionContextTokenError("bad_signature")
        payload: Mapping[str, Any] = json.loads(_decode(encoded))
        if int(payload.get("v") or 0) != 1:
            raise ExecutionContextTokenError("malformed")
        if int(payload.get("exp") or 0) < int(time.time()):
            raise ExecutionContextTokenError("expired")
        if tool_name is not None and tool_name not in set(str(value) for value in payload.get("allowed_tools") or []):
            raise ExecutionContextTokenError("tool_disallowed")
        context = payload.get("context")
        if not payload.get("task_id") or not isinstance(context, Mapping):
            raise ExecutionContextTokenError("identity_mismatch")
        decoded = ToolInvocationContext.from_mapping(context)
        extensions = dict(decoded.extensions or {})
        if extensions.get("task_id") != payload.get("task_id") or not decoded.thread_id or not decoded.run_id:
            raise ExecutionContextTokenError("identity_mismatch")
        model_settings = payload.get("model_settings")
        if not isinstance(model_settings, Mapping) or any((
            model_settings.get("context_window") != decoded.context_window,
            model_settings.get("embedding_model") != decoded.embedding_model,
            model_settings.get("llm_model") != extensions.get("llm_model"),
        )):
            raise ExecutionContextTokenError("model_context_mismatch")
        try:
            configured_context = hermes_model_context_length(required=True)
        except HermesConfigurationError as exc:
            raise ExecutionContextTokenError("model_context_mismatch") from exc
        if decoded.context_window != configured_context:
            raise ExecutionContextTokenError("model_context_mismatch")
        return decoded
    except ExecutionContextTokenError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError) as exc:
        raise ExecutionContextTokenError("malformed") from exc


def verified_token_run_id(token: str) -> str | None:
    """Return a run ID only when the token signature is authentic."""

    try:
        encoded, signature = token.split(".", 1)
        expected = _encode(hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(_decode(encoded))
        context = payload.get("context") if isinstance(payload, Mapping) else None
        run_id = context.get("run_id") if isinstance(context, Mapping) else None
        return str(run_id) if run_id else None
    except (TypeError, ValueError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError):
        return None
