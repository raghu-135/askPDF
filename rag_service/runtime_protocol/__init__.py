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
from .configuration import RuntimeConfigurationError, RuntimeEnvironment, validate_runtime_environment
from .contracts import RUNTIME_MINIMUM_COMPATIBLE_VERSION, RUNTIME_PROTOCOL_VERSION

__all__ = [
    "CANONICAL_RUNTIME_EVENT_KINDS",
    "TERMINAL_RUNTIME_EVENT_KINDS",
    "iter_sse",
    "json_envelope",
    "sse_encode",
    "structured_error",
    "validate_event_mapping",
    "RuntimeConfigurationError",
    "RuntimeEnvironment",
    "validate_runtime_environment",
    "RUNTIME_PROTOCOL_VERSION",
    "RUNTIME_MINIMUM_COMPATIBLE_VERSION",
]
