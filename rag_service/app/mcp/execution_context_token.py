"""Short-lived signed execution context for MCP clients without metadata support."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import math
import os
import time
from typing import Any, Mapping

from app.tools.context import ToolInvocationContext
from app.runtime.hermes_config import HermesConfigurationError, hermes_model_context_length


TOKEN_ARGUMENT = "_askpdf_context_token"
TOKEN_HEADER = "x-askpdf-execution-context"
TOKEN_AUDIENCE = "askpdf-mcp"
TOKEN_CLOCK_SKEW_SECONDS = 30
MAX_TOKEN_LENGTH = 65_536


class ExecutionContextTokenError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__("Invalid MCP execution context")
        self.reason = reason


def _secret() -> bytes:
    value = os.getenv("MCP_EXECUTION_CONTEXT_SECRET", "").encode()
    if len(value) < 32:
        raise ValueError("MCP_EXECUTION_CONTEXT_SECRET must contain at least 32 characters")
    return value


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def execution_context_ttl_seconds(
    limits: Mapping[str, Any] | None,
    *,
    max_duration_seconds: int | None = None,
) -> int:
    """Bound a grant to the admitted execution plus terminal confirmation."""

    values = dict(limits or {})
    duration_ms = values.get("max_duration_ms") or values.get("max_active_runtime_ms")
    if duration_ms is not None:
        duration = math.ceil(float(duration_ms) / 1000)
    elif max_duration_seconds is not None:
        duration = int(max_duration_seconds)
    else:
        duration = math.ceil(float(os.environ["AGENT_RUNTIME_READ_TIMEOUT_SECONDS"]))
    grace = math.ceil(float(os.environ["AGENT_RUNTIME_TERMINAL_CONFIRM_TIMEOUT_SECONDS"]))
    if duration <= 0 or grace <= 0:
        raise ValueError("MCP execution context duration and confirmation grace must be positive")
    return duration + grace


def issue_execution_context_token(
    context: ToolInvocationContext,
    *,
    task_id: str,
    allowed_tools: list[str],
    ttl_seconds: int = 3600,
    runtime: str = "hermes",
) -> str:
    runtime = str(runtime or "").strip().lower()
    if runtime not in {"hermes", "langgraph"}:
        raise ValueError("runtime must be hermes or langgraph")
    if ttl_seconds <= 0:
        raise ValueError("ttl_seconds must be positive")
    if not context.thread_id or not context.run_id or not str(task_id).strip():
        raise ValueError("MCP execution context requires thread, run, and task identities")
    if not allowed_tools or not all(isinstance(value, str) and value.strip() for value in allowed_tools):
        raise ValueError("MCP execution context requires a non-empty tool allowlist")
    canonical_allowed_tools = sorted({value.strip() for value in allowed_tools})
    issued_at = int(time.time())
    context_data = context.as_dict()
    extensions = dict(context_data.get("extensions") or {})
    extensions["task_id"] = task_id
    context_data["extensions"] = extensions
    payload = {
        "v": 1,
        "aud": TOKEN_AUDIENCE,
        "iat": issued_at,
        "exp": issued_at + ttl_seconds,
        "run_id": context.run_id,
        "thread_id": context.thread_id,
        "task_id": task_id,
        "runtime": runtime,
        "allowed_tools": canonical_allowed_tools,
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


def decode_execution_context_grant(
    token: str,
    *,
    tool_name: str | None = None,
) -> tuple[ToolInvocationContext, frozenset[str]]:
    try:
        if len(token) > MAX_TOKEN_LENGTH:
            raise ExecutionContextTokenError("malformed")
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
        if payload.get("aud") != TOKEN_AUDIENCE:
            raise ExecutionContextTokenError("wrong_audience")
        now = int(time.time())
        issued_at = int(payload.get("iat") or 0)
        expires_at = int(payload.get("exp") or 0)
        if issued_at <= 0 or expires_at <= issued_at or issued_at > now + TOKEN_CLOCK_SKEW_SECONDS:
            raise ExecutionContextTokenError("malformed")
        if expires_at <= now:
            raise ExecutionContextTokenError("expired")
        allowed_tools = payload.get("allowed_tools")
        if (
            not isinstance(allowed_tools, list)
            or not allowed_tools
            or not all(
                isinstance(value, str)
                and value
                and value == value.strip()
                and not any(character.isspace() for character in value)
                for value in allowed_tools
            )
        ):
            raise ExecutionContextTokenError("malformed")
        if tool_name is not None and tool_name not in set(allowed_tools):
            raise ExecutionContextTokenError("tool_disallowed")
        context = payload.get("context")
        if not payload.get("task_id") or not isinstance(context, Mapping):
            raise ExecutionContextTokenError("identity_mismatch")
        decoded = ToolInvocationContext.from_mapping(context)
        extensions = dict(decoded.extensions or {})
        if any((
            extensions.get("task_id") != payload.get("task_id"),
            decoded.run_id != payload.get("run_id"),
            decoded.thread_id != payload.get("thread_id"),
            not decoded.thread_id,
            not decoded.run_id,
        )):
            raise ExecutionContextTokenError("identity_mismatch")
        model_settings = payload.get("model_settings")
        if not isinstance(model_settings, Mapping) or any((
            model_settings.get("context_window") != decoded.context_window,
            model_settings.get("embedding_model") != decoded.embedding_model,
            model_settings.get("llm_model") != extensions.get("llm_model"),
        )):
            raise ExecutionContextTokenError("model_context_mismatch")
        runtime = str(payload.get("runtime") or "").strip().lower()
        if runtime not in {"hermes", "langgraph"}:
            raise ExecutionContextTokenError("malformed")
        if runtime == "hermes":
            try:
                configured_context = hermes_model_context_length(required=True)
            except HermesConfigurationError as exc:
                raise ExecutionContextTokenError("model_context_mismatch") from exc
            if decoded.context_window != configured_context:
                raise ExecutionContextTokenError("model_context_mismatch")
        return decoded, frozenset(allowed_tools)
    except ExecutionContextTokenError:
        raise
    except (TypeError, ValueError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError) as exc:
        raise ExecutionContextTokenError("malformed") from exc


def decode_execution_context_token(token: str, *, tool_name: str | None = None) -> ToolInvocationContext:
    context, _allowed_tools = decode_execution_context_grant(token, tool_name=tool_name)
    return context


def verified_token_run_id(token: str) -> str | None:
    """Return a run ID only when the token signature is authentic."""

    try:
        encoded, signature = token.split(".", 1)
        expected = _encode(hmac.new(_secret(), encoded.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(_decode(encoded))
        if not isinstance(payload, Mapping) or payload.get("aud") != TOKEN_AUDIENCE:
            return None
        context = payload.get("context") if isinstance(payload, Mapping) else None
        run_id = context.get("run_id") if isinstance(context, Mapping) else None
        return str(run_id) if run_id else None
    except (TypeError, ValueError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError):
        return None
