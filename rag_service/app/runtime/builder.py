"""Framework-neutral builder-provider contracts.

Builders describe and validate concrete agent definitions. They do not own
canonical product persistence or runtime execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping, Protocol

from app.runtime.adapter import AgentRuntimeEventSink
from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    RuntimeValidationResult,
)


@dataclass(frozen=True)
class BuilderCapabilities:
    """Capabilities exposed by a concrete builder provider."""

    framework: str
    builder_id: str
    schema_versions: tuple[int, ...] = (1,)
    validation: bool = True
    normalization: bool = True
    catalog: bool = True
    source: bool = True
    authoring: bool = False
    transient_tests: bool = False
    runtime_capabilities: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "builder_id": self.builder_id,
            "schema_versions": list(self.schema_versions),
            "validation": self.validation,
            "normalization": self.normalization,
            "catalog": self.catalog,
            "source": self.source,
            "authoring": self.authoring,
            "transient_tests": self.transient_tests,
            "runtime_capabilities": dict(self.runtime_capabilities),
        }


@dataclass(frozen=True)
class BuilderCatalog:
    """Neutral envelope around provider-owned catalog payloads."""

    framework: str
    builder_id: str
    schema_versions: tuple[int, ...] = (1,)
    capabilities: BuilderCapabilities | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "builder_id": self.builder_id,
            "schema_versions": list(self.schema_versions),
            "capabilities": self.capabilities.to_dict() if self.capabilities else None,
            "payload": dict(self.payload),
        }


class UnsupportedRequestOverrideError(ValueError):
    """Raised when an explicit request uses overrides a provider does not own."""

    def __init__(self, keys: list[str] | tuple[str, ...] | set[str]):
        self.keys = tuple(sorted(str(key) for key in keys))
        super().__init__(f"Unsupported request overrides: {', '.join(self.keys)}")


class AgentBuilderProvider(Protocol):
    framework: str
    builder_id: str

    async def capabilities(self, definition: AgentDefinition) -> BuilderCapabilities: ...

    async def validate(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> RuntimeValidationResult: ...

    async def normalize(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...

    def filter_request_overrides(
        self,
        definition: AgentDefinition,
        overrides: Mapping[str, Any] | None,
        *,
        reject_unsupported: bool,
    ) -> Mapping[str, Any]: ...

    async def resolve(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        thread_settings: Mapping[str, Any] | None = None,
        request_overrides: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...

    async def catalog(self, definition: AgentDefinition | None = None) -> BuilderCatalog: ...

    async def source(self, definition_id: str) -> Mapping[str, Any]: ...

    async def transient_test(
        self,
        request: AgentRuntimeRequest,
        *,
        context: Any = None,
        event_sink: AgentRuntimeEventSink | None = None,
    ) -> AgentRuntimeResult: ...

    async def resume_transient_test(
        self,
        request: AgentRuntimeRequest,
        *,
        context: Any = None,
        event_sink: AgentRuntimeEventSink | None = None,
    ) -> AgentRuntimeResult: ...

    async def cleanup_transient_test(self, request: AgentRuntimeRequest) -> Any: ...
