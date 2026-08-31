"""Framework-neutral builder-provider contracts.

Builders describe and validate concrete agent definitions. They do not own
canonical product persistence or runtime execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping, Protocol

from app.runtime.adapter import AgentRuntimeEventSink
from runtime_protocol.contracts import (
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
    capabilities: BuilderCapabilities | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "builder_id": self.builder_id,
            "capabilities": self.capabilities.to_dict() if self.capabilities else None,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True)
class BuilderTestContext:
    run: Any
    test_request: Any
    embedding_model: str
    builder_session_id: str
    resume_decision: Mapping[str, Any] | None = None


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

    def supports_task_web_search(self, definition: AgentDefinition) -> bool: ...

    def task_configuration_fields(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], ...]: ...

    def normalize_task_limits(
        self,
        limits: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...

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
