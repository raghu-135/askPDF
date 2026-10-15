"""Classify provider LLM errors for runtime events and UI."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional


def _extract_http_status_code(err_str: str) -> Optional[int]:
    patterns = (
        r"status(?:_code)?[=:]\s*(\d{3})",
        r"error code:\s*(\d{3})",
        r"\b(\d{3})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, err_str)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code in {408, 409, 429} or status_code >= 500

CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"

_CONTEXT_LENGTH_PATTERNS = (
    re.compile(r"context_length_exceeded", re.IGNORECASE),
    re.compile(r"maximum context length", re.IGNORECASE),
    re.compile(r"max(?:imum)?\s+tokens", re.IGNORECASE),
    re.compile(r"context window", re.IGNORECASE),
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"token limit", re.IGNORECASE),
)

_TRUNCATION_FINISH_REASONS = frozenset({"length", "max_tokens"})


def is_context_length_error(message: str) -> bool:
    text = str(message or "")
    if not text.strip():
        return False
    return any(pattern.search(text) for pattern in _CONTEXT_LENGTH_PATTERNS)


def classify_model_error_message(message: str) -> Mapping[str, Any]:
    """Return stable error metadata for provider failures."""

    text = str(message or "")
    if is_context_length_error(text):
        return {"code": CONTEXT_LENGTH_EXCEEDED, "retryable": False}
    status_code = _extract_http_status_code(text)
    if status_code is not None and not _is_retryable_http_status(status_code):
        return {"code": "provider_request_rejected", "retryable": False, "http_status_code": status_code}
    if status_code is not None and _is_retryable_http_status(status_code):
        return {
            "code": "provider_transient_error",
            "retryable": True,
            "http_status_code": status_code,
        }
    return {"code": "model_call_failed", "retryable": False}


def output_was_truncated(finish_reason: Any) -> bool:
    return str(finish_reason or "").strip().lower() in _TRUNCATION_FINISH_REASONS
