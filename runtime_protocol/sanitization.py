"""Bounded, secret-safe JSON values for the runtime wire boundary."""

from __future__ import annotations

import json
from typing import Any


RUNTIME_PREVIEW_LIMIT = 900
RUNTIME_REDACTED_VALUE = "[redacted]"
_SENSITIVE_PARTS = {
    "api_key", "apikey", "authorization", "bearer", "cookie", "id_token",
    "password", "private_key", "refresh_token", "resume_token", "secret",
    "set_cookie", "token",
}
_USAGE_TOKEN_PARTS = {
    "cached_tokens", "completion_tokens", "input_tokens", "output_tokens",
    "prompt_tokens", "reasoning_tokens", "token_count", "token_counts",
    "token_usage", "total_tokens",
}


def _normalized_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _sensitive(key: Any) -> bool:
    normalized = _normalized_key(key)
    if any(part in normalized for part in _USAGE_TOKEN_PARTS):
        return False
    return any(part in normalized for part in _SENSITIVE_PARTS)


def bounded_value(value: Any, *, key: Any = None) -> Any:
    """Return a JSON-safe value with secrets redacted and collection sizes bounded."""

    if key is not None and _sensitive(key):
        return RUNTIME_REDACTED_VALUE
    if value in (None, "", [], {}):
        return value
    if isinstance(value, str):
        return value[:RUNTIME_PREVIEW_LIMIT]
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [bounded_value(item) for item in value[:50]]
    if isinstance(value, dict):
        return {
            str(item_key): bounded_value(item, key=item_key)
            for item_key, item in list(value.items())[:100]
            if item not in (None, "", [], {})
        }
    try:
        return json.loads(json.dumps(value, default=str, ensure_ascii=True))
    except Exception:
        return str(value)
