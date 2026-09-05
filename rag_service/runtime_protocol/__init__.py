"""Framework-neutral wire protocol helpers shared by runtime services."""

from .protocol import (
    CANONICAL_RUNTIME_EVENT_KINDS,
    TERMINAL_RUNTIME_EVENT_KINDS,
    iter_sse,
    json_envelope,
    sse_encode,
    structured_error,
    validate_event_mapping,
    versioned_payload,
)
from .configuration import RuntimeConfigurationError, RuntimeEnvironment, parse_bounded_ratio, validate_runtime_environment
from .contracts import (
    RUNTIME_MINIMUM_COMPATIBLE_VERSION,
    RUNTIME_PROTOCOL_VERSION,
    protocol_error_details,
    require_protocol_fields,
)
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
from .validation import RuntimeProtocolValidationError, validate_runtime_result_envelope

__all__ = [
    "CANONICAL_RUNTIME_EVENT_KINDS",
    "TERMINAL_RUNTIME_EVENT_KINDS",
    "iter_sse",
    "json_envelope",
    "protocol_error_details",
    "require_protocol_fields",
    "sse_encode",
    "structured_error",
    "validate_event_mapping",
    "versioned_payload",
    "RuntimeConfigurationError",
    "RuntimeEnvironment",
    "parse_bounded_ratio",
    "validate_runtime_environment",
    "RUNTIME_PROTOCOL_VERSION",
    "RUNTIME_MINIMUM_COMPATIBLE_VERSION",
    "ToolError",
    "ToolErrorCode",
    "ToolMetrics",
    "ToolResult",
    "ToolTrace",
    "ToolWarningCode",
    "normalize_tool_result",
    "validate_tool_result_payload",
    "RuntimeProtocolValidationError",
    "validate_runtime_result_envelope",
    "MAX_TOOL_RESULT_BYTES",
    "MAX_TOOL_RESULT_COLLECTION_ITEMS",
    "MAX_TOOL_RESULT_STRING_LENGTH",
]
