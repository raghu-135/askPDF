"""Required operational limits shared by runtime and product transports."""

from __future__ import annotations

import os


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

