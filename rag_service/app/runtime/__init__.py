"""Framework-neutral runtime contracts and catalog projections."""

from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeCapabilities,
    RuntimeValidationIssue,
    RuntimeValidationResult,
    RuntimeArtifact,
    RuntimeTaskContext,
)
from app.runtime.adapter import AgentRuntimeEventSink
from app.runtime.errors import RuntimeError
from app.runtime.builder import AgentBuilderProvider, BuilderCapabilities, BuilderCatalog
from app.runtime.builder_registry import BuilderRegistry, BuilderSelectionError, builder_for_definition

__all__ = [
    "AgentDefinition",
    "AgentRuntimeEvent",
    "AgentRuntimeRequest",
    "AgentRuntimeResult",
    "ContinuationBinding",
    "RuntimeCapabilities",
    "RuntimeValidationIssue",
    "RuntimeValidationResult",
    "RuntimeArtifact",
    "RuntimeTaskContext",
    "AgentRuntimeEventSink",
    "RuntimeError",
    "AgentBuilderProvider",
    "BuilderCapabilities",
    "BuilderCatalog",
    "BuilderRegistry",
    "BuilderSelectionError",
    "builder_for_definition",
]
