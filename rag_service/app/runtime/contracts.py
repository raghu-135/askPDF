"""Stable, framework-neutral contracts for agent execution.

This module deliberately contains no LangGraph, LangChain, or runtime-specific
imports. Values are JSON-compatible so the same contracts can be used by an
in-process adapter and a future internal runtime service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional


CONTRACT_VERSION = 1
RUNTIME_OPERATION_EVENT_KINDS = frozenset({
    "operation.started",
    "operation.completed",
    "operation.failed",
    "operation.skipped",
})


class RuntimeOperationId(str, Enum):
    """Stable identifiers for the currently exposed runtime operations."""

    RUN_START = "run.start"
    RUN_GET = "run.get"
    RUN_LIST = "run.list"
    RUN_WAIT = "run.wait"
    RUN_STREAM_EVENTS = "run.stream_events"
    RUN_RESUME = "run.resume"
    RUN_CANCEL = "run.cancel"
    RUN_PAUSE = "run.pause"
    RUN_RETRY = "run.retry"
    RUN_EVENTS = "run.events"
    RUN_INSPECT_STATE = "run.inspect_state"
    RUN_APPROVAL_RESPOND = "run.approval.respond"
    INTERRUPT_RESPOND = "interrupt.respond"
    RUN_SEND_FOLLOWUP = "run.send_followup"
    RUN_INTERRUPT_WITH_INPUT = "run.interrupt_with_input"
    RUN_STEER_LIVE = "run.steer_live"
    RUN_UPDATE_STATE = "run.update_state"
    RUN_REPLAY = "run.replay"
    RUN_FORK = "run.fork"
    SUBAGENT_LIST = "subagent.list"
    SUBAGENT_SEND = "subagent.send"
    SUBAGENT_CANCEL = "subagent.cancel"
    ARTIFACT_LIST = "artifact.list"
    RUN_CONTINUE = "run.continue"
    RUN_CONTINUATION_CLEANUP = "run.continuation.cleanup"
    TRACE_PROJECT = "trace.project"


class RuntimeSupportLevel(str, Enum):
    NATIVE = "native"
    EMULATED = "emulated"
    CONDITIONAL = "conditional"
    UNSUPPORTED = "unsupported"


def _dict(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


@dataclass(frozen=True)
class AgentDefinition:
    definition_id: str
    framework: str
    builder_id: str
    category: Optional[str] = None
    display_name: Optional[str] = None
    capabilities: Mapping[str, Any] = field(default_factory=dict)
    definition_version: Optional[str] = None
    contract_version: int = CONTRACT_VERSION
    runtime_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuationBinding:
    """Opaque runtime-owned continuation state."""

    binding_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    binding_version: int = 1
    runtime_version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentRuntimeRequest:
    run_id: str
    thread_id: str
    definition_id: str
    framework: str
    builder_id: str
    input: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    continuation: Optional[ContinuationBinding] = None
    trace_id: Optional[str] = None
    authentication: Mapping[str, Any] = field(default_factory=dict)
    permissions: Mapping[str, Any] = field(default_factory=dict)
    contract_version: int = CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        if self.continuation is not None:
            value["continuation"] = self.continuation.to_dict()
        return value


@dataclass(frozen=True)
class AgentRuntimeEvent:
    event_id: str
    run_id: str
    sequence: int
    kind: str
    attempt: int = 1
    payload: Mapping[str, Any] = field(default_factory=dict)
    occurred_at: Optional[str] = None
    terminal: bool = False
    trace_id: Optional[str] = None
    runtime_version: Optional[str] = None
    continuation: Optional[ContinuationBinding] = None
    contract_version: int = CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        if self.continuation is not None:
            value["continuation"] = self.continuation.to_dict()
        return value


@dataclass(frozen=True)
class RuntimeOperationDescriptor:
    support: RuntimeSupportLevel
    enabled: bool
    disabled_reason: Optional[str] = None
    modes: tuple[str, ...] = ()
    semantics: Optional[str] = None
    confirmation: Optional[str] = None
    terminal_states: tuple[str, ...] = ()
    preserves_run_id: Optional[bool] = None
    preserves_session_id: Optional[bool] = None

    def __post_init__(self) -> None:
        if not isinstance(self.support, RuntimeSupportLevel):
            raise ValueError("support must be a RuntimeSupportLevel")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        if self.enabled and self.disabled_reason is not None:
            raise ValueError("enabled operation descriptors cannot have disabled_reason")
        if not self.enabled and not self.disabled_reason:
            raise ValueError("disabled operation descriptors require disabled_reason")
        if self.support is RuntimeSupportLevel.UNSUPPORTED and self.enabled:
            raise ValueError("unsupported operation descriptors cannot be enabled")
        for values, field_name in ((self.modes, "modes"), (self.terminal_states, "terminal_states")):
            if not isinstance(values, tuple) or not all(isinstance(item, str) and item for item in values):
                raise ValueError(f"{field_name} must contain non-empty strings")
        for value, field_name in ((self.semantics, "semantics"), (self.confirmation, "confirmation")):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string or null")
        for value, field_name in ((self.preserves_run_id, "preserves_run_id"), (self.preserves_session_id, "preserves_session_id")):
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{field_name} must be a bool or null")

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "support": self.support.value,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason,
        }
        if self.modes:
            value["modes"] = list(self.modes)
        if self.semantics is not None:
            value["semantics"] = self.semantics
        if self.confirmation is not None:
            value["confirmation"] = self.confirmation
        if self.terminal_states:
            value["terminal_states"] = list(self.terminal_states)
        if self.preserves_run_id is not None:
            value["preserves_run_id"] = self.preserves_run_id
        if self.preserves_session_id is not None:
            value["preserves_session_id"] = self.preserves_session_id
        return value


@dataclass(frozen=True)
class RuntimeCapabilities:
    operations: Mapping[str, RuntimeOperationDescriptor] = field(default_factory=dict)
    runtime_version: Optional[str] = None
    contract_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for operation, descriptor in self.operations.items():
            if not isinstance(operation, str) or not operation.strip():
                raise ValueError("capability operation identifiers must be non-empty strings")
            if not isinstance(descriptor, RuntimeOperationDescriptor):
                raise TypeError("capability operations must contain RuntimeOperationDescriptor values")

    def to_dict(self) -> Dict[str, Any]:
        ordered_operations = sorted(
            self.operations.items(),
            key=lambda item: item[0].value if isinstance(item[0], RuntimeOperationId) else str(item[0]),
        )
        return {
            "operations": {
                operation.value if isinstance(operation, RuntimeOperationId) else str(operation): descriptor.to_dict()
                for operation, descriptor in ordered_operations
            },
            "runtime_version": self.runtime_version,
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class RuntimeApprovalResponse:
    choice: str
    resolve_all: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeSteeringInput:
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeValidationIssue:
    code: str
    message: str
    path: Optional[str] = None
    severity: str = "error"
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeValidationResult:
    valid: bool
    issues: tuple[RuntimeValidationIssue, ...] = ()
    normalized_spec: Optional[Mapping[str, Any]] = None
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)
    contract_version: int = CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["issues"] = [issue.to_dict() for issue in self.issues]
        return value


@dataclass(frozen=True)
class AgentRuntimeResult:
    status: str
    output: Any = None
    clarification: Optional[Mapping[str, Any]] = None
    interruption: Optional[Mapping[str, Any]] = None
    artifacts: tuple[Mapping[str, Any], ...] = ()
    usage: Mapping[str, Any] = field(default_factory=dict)
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)
    continuation: Optional[ContinuationBinding] = None
    error: Optional[Mapping[str, Any]] = None
    contract_version: int = CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["artifacts"] = [dict(item) for item in self.artifacts]
        if self.continuation is not None:
            value["continuation"] = self.continuation.to_dict()
        return value


@dataclass(frozen=True)
class RuntimeArtifact:
    """Bounded artifact output; persistence remains a control-plane concern."""

    kind: str
    content: Optional[str] = None
    artifact_id: Optional[str] = None
    sha256: Optional[str] = None
    media_type: str = "text/plain"
    todo_id: Optional[str] = None
    subagent_run_id: Optional[str] = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    source_refs: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimeTaskContext:
    """Read-only product context supplied to a runtime task invocation."""

    task_id: str
    objective: str = ""
    todos: tuple[Mapping[str, Any], ...] = ()
    artifact_manifests: tuple[Mapping[str, Any], ...] = ()
    artifact_contents: Mapping[str, str] = field(default_factory=dict)
    limits: Mapping[str, Any] = field(default_factory=dict)
    permissions: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "todos": [dict(value) for value in self.todos],
            "artifact_manifests": [dict(value) for value in self.artifact_manifests],
            "artifact_contents": dict(self.artifact_contents),
            "limits": dict(self.limits),
            "permissions": dict(self.permissions),
        }


def mapping_copy(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a shallow JSON-compatible mapping copy for adapter boundaries."""

    return _dict(value)
