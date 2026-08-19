"""Neutral runtime adapter protocol and invocation context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Mapping, Optional, Protocol

from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeCapabilities,
    RuntimeApprovalResponse,
    RuntimeSteeringInput,
    RuntimeValidationResult,
)


@dataclass(frozen=True)
class RuntimeExecutionContext:
    """In-process compatibility inputs kept outside the wire contract."""

    request: Any = None
    embedding_model: Optional[str] = None
    resolved_spec: Mapping[str, Any] = field(default_factory=dict)
    agent_run_context: Mapping[str, Any] = field(default_factory=dict)
    trace_recorder: Any = None
    cancellation_checker: Any = None
    result_projector: Any = None
    task_id: Optional[str] = None
    task_worker_id: Optional[str] = None


class AgentRuntimeEventSink(Protocol):
    async def emit(self, event: Any) -> None: ...


class AgentRuntimeAdapter(Protocol):
    framework: str
    builder_id: str

    async def capabilities(self, definition: AgentDefinition) -> RuntimeCapabilities: ...

    async def validate(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> RuntimeValidationResult: ...

    async def start(
        self,
        request: AgentRuntimeRequest,
        *,
        context: RuntimeExecutionContext,
        event_sink: AgentRuntimeEventSink | None = None,
    ) -> AgentRuntimeResult: ...

    async def resume(
        self,
        request: AgentRuntimeRequest,
        *,
        interrupt: Mapping[str, Any],
        context: RuntimeExecutionContext,
        event_sink: AgentRuntimeEventSink | None = None,
    ) -> AgentRuntimeResult: ...

    async def continue_run(
        self,
        request: AgentRuntimeRequest,
        *,
        context: RuntimeExecutionContext,
        event_sink: AgentRuntimeEventSink | None = None,
    ) -> Optional[AgentRuntimeResult]: ...

    async def cancel(self, request: AgentRuntimeRequest) -> Any: ...

    async def respond_to_approval(self, request: AgentRuntimeRequest, response: RuntimeApprovalResponse) -> Any: ...

    async def steer(self, request: AgentRuntimeRequest, steering: RuntimeSteeringInput) -> Any: ...

    async def inspect(self, request: AgentRuntimeRequest) -> Mapping[str, Any]: ...

    async def delete_continuation(self, continuation: ContinuationBinding) -> Any: ...

    async def project_trace(
        self,
        events: list[Mapping[str, Any]],
        *,
        run_id: str,
        context: RuntimeExecutionContext | None = None,
    ) -> list[Any]: ...
