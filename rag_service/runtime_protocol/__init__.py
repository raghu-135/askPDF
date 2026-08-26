"""Framework-neutral wire protocol helpers shared by runtime services."""

from .protocol import (
    CANONICAL_RUNTIME_EVENT_KINDS,
    TERMINAL_RUNTIME_EVENT_KINDS,
    iter_sse,
    json_envelope,
    sse_encode,
    structured_error,
    validate_event_mapping,
)

__all__ = [
    "CANONICAL_RUNTIME_EVENT_KINDS",
    "TERMINAL_RUNTIME_EVENT_KINDS",
    "iter_sse",
    "json_envelope",
    "sse_encode",
    "structured_error",
    "validate_event_mapping",
]
