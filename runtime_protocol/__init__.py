"""Framework-neutral wire protocol helpers shared by runtime services."""

from .protocol import (
    CANONICAL_RUNTIME_EVENT_KINDS,
    TERMINAL_RUNTIME_EVENT_KINDS,
    iter_sse,
    json_envelope,
    json_payload,
    sse_encode,
    structured_error,
    validate_event_mapping,
)
from .configuration import RuntimeConfigurationError, RuntimeEnvironment, parse_bounded_ratio, validate_runtime_environment
from .auth import PUBLIC_OPERATIONAL_PATHS, bearer_token, valid_bearer_token
from .tool_contract import (
    MAX_TOOL_RESULT_BYTES,
    MAX_TOOL_RESULT_COLLECTION_ITEMS,
    MAX_TOOL_RESULT_STRING_LENGTH,
    ToolError,
    ToolErrorCode,
    ToolMetrics,
    ToolResult,
    ToolTrace,
    ToolWarningCode,
    normalize_tool_result,
    validate_tool_result_payload,
)
from .validation import (
    RUNTIME_RESULT_STATUSES,
    RuntimeProtocolValidationError,
    validate_runtime_result_envelope,
    validate_runtime_result_for_event,
)

__all__ = [
    "CANONICAL_RUNTIME_EVENT_KINDS",
    "TERMINAL_RUNTIME_EVENT_KINDS",
    "iter_sse",
    "json_envelope",
    "json_payload",
    "sse_encode",
    "structured_error",
    "validate_event_mapping",
    "RuntimeConfigurationError",
    "RuntimeEnvironment",
    "parse_bounded_ratio",
    "validate_runtime_environment",
    "PUBLIC_OPERATIONAL_PATHS",
    "bearer_token",
    "valid_bearer_token",
    "ToolError",
    "ToolErrorCode",
    "ToolMetrics",
    "ToolResult",
    "ToolTrace",
    "ToolWarningCode",
    "normalize_tool_result",
    "validate_tool_result_payload",
    "RuntimeProtocolValidationError",
    "RUNTIME_RESULT_STATUSES",
    "validate_runtime_result_envelope",
    "validate_runtime_result_for_event",
    "MAX_TOOL_RESULT_BYTES",
    "MAX_TOOL_RESULT_COLLECTION_ITEMS",
    "MAX_TOOL_RESULT_STRING_LENGTH",
]
