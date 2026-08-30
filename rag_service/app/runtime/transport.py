"""Compatibility re-exports for shared runtime wire serialization."""

from runtime_protocol.serialization import (
    WIRE_VERSION,
    ServerEnvelope,
    _binding,
    capabilities_from_dict,
    definition_from_dict,
    event_from_dict,
    iter_sse,
    json_envelope,
    request_from_dict,
    result_from_dict,
    sse_encode,
    sse_encode_payload,
    validate_wire_version,
    validation_from_dict,
)

__all__ = [
    "WIRE_VERSION", "ServerEnvelope", "capabilities_from_dict",
    "definition_from_dict", "event_from_dict", "iter_sse", "json_envelope",
    "request_from_dict", "result_from_dict", "sse_encode",
    "sse_encode_payload", "validate_wire_version", "validation_from_dict",
]
