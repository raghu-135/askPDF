"""Required operational limits shared by runtime and product transports."""

from __future__ import annotations

import os
import json
import math
from typing import Any, Mapping


MAX_RUNTIME_JSON_BYTES = 256_000
MAX_RUNTIME_JSON_DEPTH = 12
MAX_RUNTIME_JSON_COLLECTION_ITEMS = 2_000
MAX_RUNTIME_JSON_STRING_LENGTH = 20_000


def validate_bounded_json(value: Mapping[str, Any], *, field_name: str) -> dict[str, Any]:
    """Validate runtime control payloads without coercion or truncation."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    collection_items = 0
    def visit(item: Any, depth: int) -> None:
        nonlocal collection_items
        if depth > MAX_RUNTIME_JSON_DEPTH:
            raise ValueError(f"{field_name} exceeds the maximum nesting depth")
        if isinstance(item, str):
            if len(item) > MAX_RUNTIME_JSON_STRING_LENGTH:
                raise ValueError(f"{field_name} contains an oversized string")
        elif isinstance(item, Mapping):
            collection_items += len(item)
            if collection_items > MAX_RUNTIME_JSON_COLLECTION_ITEMS:
                raise ValueError(f"{field_name} contains too many collection items")
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError(f"{field_name} contains a non-string object key")
                visit(key, depth + 1)
                visit(child, depth + 1)
        elif isinstance(item, list):
            collection_items += len(item)
            if collection_items > MAX_RUNTIME_JSON_COLLECTION_ITEMS:
                raise ValueError(f"{field_name} contains too many collection items")
            for child in item:
                visit(child, depth + 1)
        elif item is None or isinstance(item, (bool, int)):
            return
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError(f"{field_name} contains a non-finite number")
        else:
            raise ValueError(f"{field_name} contains a non-JSON value")

    visit(value, 0)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    if len(encoded.encode("utf-8")) > MAX_RUNTIME_JSON_BYTES:
        raise ValueError(f"{field_name} exceeds the maximum serialized size")
    return dict(value)


def required_positive_float(name: str) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        raise RuntimeError(f"Required environment variable {name} is not configured")
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be numeric") from exc
    if value <= 0:
        raise RuntimeError(f"Environment variable {name} must be greater than zero")
    return value


def positive_float_value(value: Any, *, name: str) -> float:
    """Validate a persisted/configured positive numeric limit."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return normalized


def required_positive_int(name: str) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        raise RuntimeError(f"Required environment variable {name} is not configured")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"Environment variable {name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"Environment variable {name} must be greater than zero")
    return value
