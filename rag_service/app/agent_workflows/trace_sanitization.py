from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from app.agent_workflows.trace import compact_preview


TRACE_PREVIEW_LIMIT = 900
TRACE_REDACTED_VALUE = "[redacted]"
TRACE_SENSITIVE_KEY_PARTS = {
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "cookie",
    "id_token",
    "password",
    "private_key",
    "refresh_token",
    "resume_token",
    "secret",
    "set_cookie",
    "token",
}
TRACE_NON_SECRET_TOKEN_KEY_PARTS = {
    "cached_tokens",
    "completion_tokens",
    "input_tokens",
    "output_tokens",
    "prompt_tokens",
    "reasoning_tokens",
    "token_count",
    "token_counts",
    "token_usage",
    "total_tokens",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _clean_dict(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _normalized_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _is_sensitive_key(key: Any) -> bool:
    normalized = _normalized_key(key)
    if any(part in normalized for part in TRACE_NON_SECRET_TOKEN_KEY_PARTS):
        return False
    return any(part in normalized for part in TRACE_SENSITIVE_KEY_PARTS)


def _bounded_value(value: Any, *, key: Any = None) -> Any:
    if key is not None and _is_sensitive_key(key):
        return TRACE_REDACTED_VALUE
    if value in (None, "", [], {}):
        return value
    if isinstance(value, str):
        # Streaming deltas are content fragments; trimming either edge changes
        # the reconstructed output when adjacent chunks are coalesced.
        if _normalized_key(key) == "delta":
            return value[:TRACE_PREVIEW_LIMIT]
        return compact_preview(value, limit=TRACE_PREVIEW_LIMIT)
    if isinstance(value, list):
        return [_bounded_value(item) for item in value[:50]]
    if isinstance(value, dict):
        return {
            item_key: _bounded_value(item, key=item_key)
            for item_key, item in value.items()
            if item not in (None, "", [], {})
        }
    # Framework-owned objects (for example LangGraph GraphInterrupt values)
    # must never cross the persisted JSON boundary as live Python instances.
    return _jsonable(value)


def _jsonable(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, default=str, ensure_ascii=True))
    except Exception:
        return str(value)


def _otel_attr_value(value: Any) -> Any:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list) and all(isinstance(item, (str, bool, int, float)) for item in value):
        return value
    return json.dumps(_jsonable(value), ensure_ascii=True, sort_keys=True)


def _set_attributes(span: Any, attributes: Mapping[str, Any]) -> None:
    for key, value in attributes.items():
        otel_value = _otel_attr_value(value)
        if otel_value is not None:
            span.set_attribute(key, otel_value)
