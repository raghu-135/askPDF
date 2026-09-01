"""Neutral runtime adapter base class and JSON-only invocation context."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Mapping, NoReturn, Optional, Protocol

from runtime_protocol.contracts import (
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeApprovalResponse,
    RuntimeCapabilities,
    RuntimeSteeringInput,
    RuntimeValidationResult,
    RuntimeTaskContext,
    RuntimeOperationId,
)
from runtime_protocol.errors import RuntimeError


@dataclass(frozen=True)
class RuntimeInvocationContext:
    """Product-owned values that may be serialized onto the runtime wire."""

    request_payload: Mapping[str, Any] = field(default_factory=dict)
    embedding_model: Optional[str] = None
    resolved_spec: Mapping[str, Any] = field(default_factory=dict)
    agent_run_context: Mapping[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    task_worker_id: Optional[str] = None
    task_context: Optional[RuntimeTaskContext] = None


class AgentRuntimeEventSink(Protocol):
    async def emit_runtime_event(self, event: AgentRuntimeEvent) -> None: ...


class AgentRuntimeAdapter(ABC):
    """Universal runtime SPI with safe defaults for optional operations."""

    framework: str
    builder_id: str
    supports_task_pause: bool = False
    supports_external_task_pause: bool = False
    implemented_operations: frozenset[RuntimeOperationId] = frozenset()

    async def prepare_request(
        self,
        request: AgentRuntimeRequest,
        *,
        context: RuntimeInvocationContext,
    ) -> AgentRuntimeRequest:
        """Let a concrete adapter encode neutral context for its transport."""
        return request

    def _unsupported(self, operation_id: str, explanation: str) -> NoReturn:
        raise RuntimeError.capability_unsupported(
            operation_id=operation_id,
            framework=self.framework,
            builder_id=self.builder_id,
            explanation=explanation,
        )

    @abstractmethod
    async def capabilities(self, definition: AgentDefinition) -> RuntimeCapabilities: ...

    async def deployment_capabilities(self) -> RuntimeCapabilities:
        """Discover capabilities owned by this concrete deployment."""
        return await self.capabilities(
            AgentDefinition(
                definition_id=f"deployment:{self.framework}:{self.builder_id}",
                framework=self.framework,
                builder_id=self.builder_id,
            )
        )

    @abstractmethod
    async def validate(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> RuntimeValidationResult: ...

    @abstractmethod
    async def start(
        self,
        request: AgentRuntimeRequest,
        *,
        context: RuntimeInvocationContext,
        event_sink: AgentRuntimeEventSink | None = None,
    ) -> AgentRuntimeResult: ...

    async def get_run(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        self._unsupported("run.get", "This runtime does not expose run inspection")

    async def list_runs(
        self,
        *,
        thread_id: str,
        definition_id: str | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[Mapping[str, Any]]:
        self._unsupported("run.list", "This runtime does not expose run listing")

    async def wait(
        self,
        request: AgentRuntimeRequest,
        *,
        timeout_seconds: float | None = None,
    ) -> AgentRuntimeResult:
        self._unsupported("run.wait", "This runtime does not expose run waiting")

    async def stream_events(
        self,
        request: AgentRuntimeRequest,
        *,
        after_sequence: int = 0,
    ) -> AsyncIterator[Any]:
        self._unsupported("run.events", "This runtime does not expose independent event streaming")

    async def resume(
        self,
        request: AgentRuntimeRequest,
        *,
        interrupt: Mapping[str, Any],
        context: RuntimeInvocationContext,
        event_sink: AgentRuntimeEventSink | None = None,
    ) -> AgentRuntimeResult:
        self._unsupported("run.resume", "This runtime does not expose run resumption")

    async def continue_run(
        self,
        request: AgentRuntimeRequest,
        *,
        context: RuntimeInvocationContext,
        event_sink: AgentRuntimeEventSink | None = None,
    ) -> Optional[AgentRuntimeResult]:
        raise RuntimeError(
            "runtime_continuation_unavailable",
            "This runtime does not expose the internal approval continuation path",
        )

    async def cancel(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        self._unsupported("run.cancel", "This runtime does not expose run cancellation")

    async def pause(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        self._unsupported("task.pause", "This runtime does not expose task pausing")

    async def respond_to_approval(
        self,
        request: AgentRuntimeRequest,
        response: RuntimeApprovalResponse,
    ) -> Mapping[str, Any]:
        self._unsupported("run.approval.respond", "This runtime does not expose approval responses")

    async def send_followup(
        self,
        request: AgentRuntimeRequest,
        input: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._unsupported("run.send_followup", "This runtime does not expose queued follow-up input")

    async def interrupt_with_input(
        self,
        request: AgentRuntimeRequest,
        input: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._unsupported("run.interrupt_with_input", "This runtime does not expose interrupt-and-continue input")

    async def steer_live(self, request: AgentRuntimeRequest, steering: RuntimeSteeringInput) -> Mapping[str, Any]:
        self._unsupported("run.steer_live", "This runtime does not provide live steering")

    async def inspect_state(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        self._unsupported("run.inspect_state", "This runtime does not expose durable state inspection")

    async def replay(self, request: AgentRuntimeRequest) -> AgentRuntimeResult:
        self._unsupported("run.replay", "This runtime does not expose continuation replay")

    async def fork(self, request: AgentRuntimeRequest) -> AgentRuntimeResult:
        self._unsupported("run.fork", "This runtime does not expose continuation forks")

    async def list_subagents(self, request: AgentRuntimeRequest) -> list[Mapping[str, Any]]:
        self._unsupported("subagent.list", "This runtime does not expose subagent listing")

    async def send_to_subagent(
        self,
        request: AgentRuntimeRequest,
        subagent_id: str,
        input: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._unsupported("subagent.send", "This runtime does not expose subagent messaging")

    async def cancel_subagent(self, request: AgentRuntimeRequest, subagent_id: str) -> Mapping[str, Any]:
        self._unsupported("subagent.cancel", "This runtime does not expose subagent cancellation")

    async def list_artifacts(self, request: AgentRuntimeRequest) -> list[Mapping[str, Any]]:
        self._unsupported("artifact.list", "This runtime does not expose runtime artifacts")

    async def delete_continuation(self, continuation: ContinuationBinding) -> Any:
        self._unsupported("run.continuation.cleanup", "This runtime does not expose continuation cleanup")

    async def project_trace(
        self,
        events: list[Mapping[str, Any]],
        *,
        run_id: str,
        context: RuntimeInvocationContext | None = None,
    ) -> list[Any]:
        self._unsupported("trace.project", "This runtime does not project runtime events into product traces")
