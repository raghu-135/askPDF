"""Compatibility re-exports for the shared runtime protocol.

New code must import these contracts from :mod:`runtime_protocol.contracts`.
"""

from runtime_protocol.contracts import (
    CONTRACT_VERSION,
    RUNTIME_OPERATION_EVENT_KINDS,
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeArtifact,
    RuntimeCapabilities,
    RuntimeTaskContext,
    RuntimeValidationIssue,
    RuntimeValidationResult,
    mapping_copy,
)

__all__ = [
    "CONTRACT_VERSION", "RUNTIME_OPERATION_EVENT_KINDS", "AgentDefinition",
    "AgentRuntimeEvent", "AgentRuntimeRequest", "AgentRuntimeResult",
    "ContinuationBinding", "RuntimeArtifact", "RuntimeCapabilities",
    "RuntimeTaskContext", "RuntimeValidationIssue", "RuntimeValidationResult",
    "mapping_copy",
]
