"""Stable, framework-neutral contracts for agent execution.

This module deliberately contains no LangGraph, LangChain, or runtime-specific
imports. Values are JSON-compatible so the same contracts can be used by an
in-process adapter and a future internal runtime service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional


RUNTIME_OPERATION_EVENT_KINDS = frozenset({
    "operation.started",
    "operation.completed",
    "operation.failed",
    "operation.skipped",
})


class RuntimeEventKind(str, Enum):
    RUN_QUEUED = "run.queued"
    RUN_STARTED = "run.started"
    RUN_PAUSED = "run.paused"
    RUN_RESUMED = "run.resumed"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCEL_REQUESTED = "run.cancel_requested"
    RUN_CANCELLED = "run.cancelled"
    OUTPUT_DELTA = "output.delta"
    OUTPUT_COMPLETED = "output.completed"
    REASONING_AVAILABLE = "reasoning.available"
    INTERRUPT_REQUESTED = "interrupt.requested"
    INTERRUPT_RESPONDED = "interrupt.responded"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_RESPONDED = "approval.responded"
    TOOL_STARTED = "tool.started"
    TOOL_PROGRESS = "tool.progress"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    SUBAGENT_STARTED = "subagent.started"
    SUBAGENT_PROGRESS = "subagent.progress"
    SUBAGENT_COMPLETED = "subagent.completed"
    SUBAGENT_FAILED = "subagent.failed"
    SUBAGENT_CANCELLED = "subagent.cancelled"
    ARTIFACT_CREATED = "artifact.created"
    ARTIFACT_UPDATED = "artifact.updated"
    ARTIFACT_COMPLETED = "artifact.completed"
    RUNTIME_EVENT = "runtime.event"


CANONICAL_RUNTIME_EVENT_KINDS = frozenset(item.value for item in RuntimeEventKind) | RUNTIME_OPERATION_EVENT_KINDS
TERMINAL_RUNTIME_EVENT_KINDS = frozenset({
    RuntimeEventKind.RUN_COMPLETED.value,
    RuntimeEventKind.RUN_FAILED.value,
    RuntimeEventKind.RUN_CANCELLED.value,
})
RUNTIME_EVENT_FAMILY_PREFIXES = (
    "run.", "output.", "reasoning.", "interrupt.", "approval.",
    "tool.", "subagent.", "artifact.", "operation.",
)


class RuntimeOperationId(str, Enum):
    """Stable identifiers for the currently exposed runtime operations."""

    RUN_START = "run.start"
    RUN_GET = "run.get"
    RUN_LIST = "run.list"
    RUN_WAIT = "run.wait"
    RUN_RESUME = "run.resume"
    RUN_CANCEL = "run.cancel"
    RUN_EVENTS = "run.events"
    RUN_INSPECT_STATE = "run.inspect_state"
    RUN_APPROVAL_RESPOND = "run.approval.respond"
    RUN_SEND_FOLLOWUP = "run.send_followup"
    RUN_INTERRUPT_WITH_INPUT = "run.interrupt_with_input"
    RUN_STEER_LIVE = "run.steer_live"
    RUN_UPDATE_STATE = "run.update_state"
    RUN_REPLAY = "run.replay"
    RUN_FORK = "run.fork"
    TASK_START = "task.start"
    TASK_PAUSE = "task.pause"
    TASK_RESUME = "task.resume"
    TASK_CANCEL = "task.cancel"
    TASK_RETRY = "task.retry"
    SUBAGENT_LIST = "subagent.list"
    SUBAGENT_SEND = "subagent.send"
    SUBAGENT_CANCEL = "subagent.cancel"
    ARTIFACT_LIST = "artifact.list"
    RUN_CONTINUATION_CLEANUP = "run.continuation.cleanup"
    TRACE_PROJECT = "trace.project"


class RuntimeSupportLevel(str, Enum):
    NATIVE = "native"
    EMULATED = "emulated"
    CONDITIONAL = "conditional"
    UNSUPPORTED = "unsupported"


class RuntimeOperationOwner(str, Enum):
    PRODUCT = "product"
    RUNTIME = "runtime"


class RuntimeCapabilityDisabledReason(str, Enum):
    RUNTIME_CAPABILITY_UNSUPPORTED = "runtime_capability_unsupported"
    RUNTIME_CAPABILITY_UNAVAILABLE = "runtime_capability_unavailable"
    RUNTIME_CONFIGURATION_INVALID = "runtime_configuration_invalid"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    DEFINITION_CAPABILITY_UNAVAILABLE = "definition_capability_unavailable"
    DEFINITION_POLICY = "definition_policy"
    DEFINITION_NOT_TASK_RUNTIME = "definition_not_task_runtime"
    ADAPTER_OPERATION_UNMAPPED = "adapter_operation_unmapped"
    ADAPTER_OPERATION_UNIMPLEMENTED = "adapter_operation_unimplemented"
    CHECKPOINT_STORE_UNAVAILABLE = "checkpoint_store_unavailable"
    TASK_RUN_NOT_CREATED = "task_run_not_created"
    RUN_ALREADY_CREATED = "run_already_created"
    TASK_ALREADY_STARTED = "task_already_started"
    RUN_TERMINAL = "run_terminal"
    NO_PENDING_INTERRUPT = "no_pending_interrupt"
    TASK_NOT_PAUSEABLE = "task_not_pauseable"
    TASK_NOT_RESUMABLE = "task_not_resumable"
    TASK_NOT_RETRYABLE = "task_not_retryable"
    TASK_TERMINAL = "task_terminal"
    RUNTIME_BINDING_UNAVAILABLE = "runtime_binding_unavailable"
    RUN_NOT_CHECKPOINT_BOUNDARY = "run_not_checkpoint_boundary"


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
    definition_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContinuationBinding:
    """Opaque runtime-owned continuation state."""

    binding_type: str
    payload: Mapping[str, Any] = field(default_factory=dict)

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
    source_metadata: Mapping[str, Any] = field(default_factory=dict)
    continuation: Optional[ContinuationBinding] = None

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        if self.continuation is not None:
            value["continuation"] = self.continuation.to_dict()
        return value


@dataclass(frozen=True)
class RuntimeOperationDescriptor:
    support: RuntimeSupportLevel
    owner: RuntimeOperationOwner
    enabled: bool
    disabled_reason: Optional[RuntimeCapabilityDisabledReason] = None
    modes: tuple[str, ...] = ()
    semantics: Optional[str] = None
    confirmation: Optional[str] = None
    terminal_states: tuple[str, ...] = ()
    preserves_run_id: Optional[bool] = None
    preserves_session_id: Optional[bool] = None
    requires_runtime_binding: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.support, RuntimeSupportLevel):
            raise ValueError("support must be a RuntimeSupportLevel")
        if not isinstance(self.owner, RuntimeOperationOwner):
            raise ValueError("owner must be a RuntimeOperationOwner")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        if self.enabled and self.disabled_reason is not None:
            raise ValueError("enabled operation descriptors cannot have disabled_reason")
        if not self.enabled and not self.disabled_reason:
            raise ValueError("disabled operation descriptors require disabled_reason")
        if self.disabled_reason is not None and not isinstance(
            self.disabled_reason, RuntimeCapabilityDisabledReason
        ):
            raise ValueError("disabled_reason must be a RuntimeCapabilityDisabledReason")
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
        if not isinstance(self.requires_runtime_binding, bool):
            raise TypeError("requires_runtime_binding must be a bool")

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "support": self.support.value,
            "owner": self.owner.value,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason.value if self.disabled_reason else None,
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
        if self.requires_runtime_binding:
            value["requires_runtime_binding"] = True
        return value


@dataclass(frozen=True)
class RuntimeFeatureDescriptor:
    """Typed descriptive capability for runtime-provided harness features."""

    support: RuntimeSupportLevel
    enabled: bool
    disabled_reason: Optional[RuntimeCapabilityDisabledReason] = None
    semantics: Optional[str] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.support, RuntimeSupportLevel):
            raise ValueError("support must be a RuntimeSupportLevel")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool")
        if self.enabled and self.disabled_reason is not None:
            raise ValueError("enabled feature descriptors cannot have disabled_reason")
        if not self.enabled and not self.disabled_reason:
            raise ValueError("disabled feature descriptors require disabled_reason")
        if self.disabled_reason is not None and not isinstance(
            self.disabled_reason, RuntimeCapabilityDisabledReason
        ):
            raise ValueError("disabled_reason must be a RuntimeCapabilityDisabledReason")
        if self.support is RuntimeSupportLevel.UNSUPPORTED and self.enabled:
            raise ValueError("unsupported feature descriptors cannot be enabled")
        if self.semantics is not None and not isinstance(self.semantics, str):
            raise ValueError("semantics must be a string or null")

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "support": self.support.value,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason.value if self.disabled_reason else None,
        }
        if self.semantics is not None:
            value["semantics"] = self.semantics
        if self.details:
            value["details"] = dict(self.details)
        return value


@dataclass(frozen=True)
class RuntimeCapabilities:
    operations: Mapping[RuntimeOperationId, RuntimeOperationDescriptor] = field(default_factory=dict)
    features: Mapping[str, RuntimeFeatureDescriptor] = field(default_factory=dict)
    deployment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for operation, descriptor in self.operations.items():
            if not isinstance(operation, RuntimeOperationId):
                raise ValueError("capability operation identifiers must be RuntimeOperationId values")
            if not isinstance(descriptor, RuntimeOperationDescriptor):
                raise TypeError("capability operations must contain RuntimeOperationDescriptor values")
        for feature, descriptor in self.features.items():
            if not isinstance(feature, str) or not feature.strip():
                raise ValueError("capability feature identifiers must be non-empty strings")
            if not isinstance(descriptor, RuntimeFeatureDescriptor):
                raise TypeError("capability features must contain RuntimeFeatureDescriptor values")

    def to_dict(self) -> Dict[str, Any]:
        ordered_operations = sorted(self.operations.items(), key=lambda item: item[0].value)
        return {
            "operations": {
                operation.value: descriptor.to_dict()
                for operation, descriptor in ordered_operations
            },
            "features": {
                str(feature): descriptor.to_dict()
                for feature, descriptor in sorted(self.features.items(), key=lambda item: item[0])
            },
            "deployment": dict(self.deployment),
        }


def native(
    *,
    owner: RuntimeOperationOwner = RuntimeOperationOwner.RUNTIME,
    enabled: bool = True,
    disabled_reason: RuntimeCapabilityDisabledReason | None = None,
    **kwargs: Any,
) -> RuntimeOperationDescriptor:
    return RuntimeOperationDescriptor(
        RuntimeSupportLevel.NATIVE,
        owner,
        enabled,
        disabled_reason=disabled_reason,
        **kwargs,
    )


def conditional(
    *,
    owner: RuntimeOperationOwner = RuntimeOperationOwner.RUNTIME,
    enabled: bool,
    disabled_reason: RuntimeCapabilityDisabledReason | None = None,
    **kwargs: Any,
) -> RuntimeOperationDescriptor:
    return RuntimeOperationDescriptor(
        RuntimeSupportLevel.CONDITIONAL,
        owner,
        enabled,
        disabled_reason=disabled_reason,
        **kwargs,
    )


def emulated(
    *,
    owner: RuntimeOperationOwner = RuntimeOperationOwner.RUNTIME,
    enabled: bool = True,
    disabled_reason: RuntimeCapabilityDisabledReason | None = None,
    **kwargs: Any,
) -> RuntimeOperationDescriptor:
    return RuntimeOperationDescriptor(
        RuntimeSupportLevel.EMULATED,
        owner,
        enabled,
        disabled_reason=disabled_reason,
        **kwargs,
    )


def unsupported(
    *,
    owner: RuntimeOperationOwner = RuntimeOperationOwner.RUNTIME,
    disabled_reason: RuntimeCapabilityDisabledReason = RuntimeCapabilityDisabledReason.RUNTIME_CAPABILITY_UNSUPPORTED,
    **kwargs: Any,
) -> RuntimeOperationDescriptor:
    return RuntimeOperationDescriptor(
        RuntimeSupportLevel.UNSUPPORTED,
        owner,
        False,
        disabled_reason=disabled_reason,
        **kwargs,
    )


@dataclass(frozen=True)
class RuntimeApprovalResponse:
    decision: str
    modifications: Optional[Mapping[str, Any]] = None
    feedback: Optional[str] = None
    scope: Optional[str] = None

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
