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
    RUN_CLARIFICATION = "run.clarification"
    RUN_FAILED = "run.failed"
    RUN_CANCEL_REQUESTED = "run.cancel_requested"
    RUN_CANCELLED = "run.cancelled"
    OUTPUT_DELTA = "output.delta"
    OUTPUT_COMPLETED = "output.completed"
    LLM_STARTED = "llm.started"
    LLM_COMPLETED = "llm.completed"
    LLM_FAILED = "llm.failed"
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
    DISPATCH_PLANNED = "dispatch.planned"
    DISPATCH_STARTED = "dispatch.started"
    DISPATCH_BARRIER_REACHED = "dispatch.barrier_reached"
    DISPATCH_CANCELLED = "dispatch.cancelled"
    WORKER_QUEUED = "worker.queued"
    WORKER_STARTED = "worker.started"
    WORKER_PROGRESS = "worker.progress"
    WORKER_RETRYING = "worker.retrying"
    WORKER_COMPLETED = "worker.completed"
    WORKER_SKIPPED = "worker.skipped"
    WORKER_FAILED = "worker.failed"
    WORKER_TIMED_OUT = "worker.timed_out"
    WORKER_CANCELLED = "worker.cancelled"
    AGGREGATION_COMPLETED = "aggregation.completed"
    AGGREGATION_PARTIAL = "aggregation.partial"
    RUNTIME_EVENT = "runtime.event"


class RuntimeTaskResultStatus(str, Enum):
    """Framework-neutral outcome of one agentic task execution."""

    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


CANONICAL_RUNTIME_EVENT_KINDS = frozenset(item.value for item in RuntimeEventKind) | RUNTIME_OPERATION_EVENT_KINDS
TERMINAL_RUNTIME_EVENT_KINDS = frozenset({
    RuntimeEventKind.RUN_COMPLETED.value,
    RuntimeEventKind.RUN_CLARIFICATION.value,
    RuntimeEventKind.RUN_FAILED.value,
    RuntimeEventKind.RUN_CANCELLED.value,
})
RUNTIME_EVENT_FAMILY_PREFIXES = (
    "run.", "output.", "llm.", "reasoning.", "interrupt.", "approval.",
    "tool.", "subagent.", "artifact.", "operation.", "dispatch.", "worker.", "aggregation.",
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
    TASK_RESULT_REVIEW_RESPOND = "task.result_review.respond"
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


class RuntimeFeatureId(str, Enum):
    PLANNING = "planning"
    PARALLEL_DISPATCH = "parallel_dispatch"
    ARTIFACTS = "artifacts"
    SUBAGENT_ORCHESTRATION = "subagent_orchestration"
    MEMORY = "memory"
    TOOLS = "tools"
    DELEGATION = "delegation"
    SKILLS = "skills"


class RuntimeCapabilitySemantics(str, Enum):
    PERSISTED_PRODUCT_EVENT_JOURNAL = "persisted_product_event_journal"
    PRODUCT_RUN_INSPECTION = "product_run_inspection"
    PRODUCT_RUN_LISTING = "product_run_listing"
    PRODUCT_TASK_ARTIFACT_LISTING = "product_task_artifact_listing"
    PRODUCT_TASK_START = "product_task_start"
    PRODUCT_TASK_PAUSE = "product_task_pause"
    PRODUCT_TASK_RESUME = "product_task_resume"
    PRODUCT_TASK_CANCEL = "product_task_cancel"
    PRODUCT_TASK_RETRY = "product_task_retry"
    PRODUCT_TASK_RESULT_REVIEW = "product_task_result_review"
    RESUME_FROM_INTERRUPT = "resume_from_interrupt"
    CHECKPOINT_STATE_INSPECTION = "checkpoint_state_inspection"
    CHECKPOINT_BOUNDARY_UPDATE = "checkpoint_boundary_update"
    CHECKPOINT_THREAD_CLEANUP = "checkpoint_thread_cleanup"
    DEFINITION_PLANNER_NODES = "definition_planner_nodes"
    DEFINITION_PARALLEL_DISPATCH = "definition_parallel_dispatch"
    DEFINITION_ARTIFACT_POLICY = "definition_artifact_policy"
    PRODUCT_MANAGED_SUBAGENTS = "product_managed_subagents"
    DEFINITION_TOOL_POLICY = "definition_tool_policy"


class RuntimeCancellationMode(str, Enum):
    INTERRUPT = "interrupt"
    COOPERATIVE = "cooperative"


class RuntimeConfirmationMode(str, Enum):
    ASYNCHRONOUS = "asynchronous"
    BOUNDED = "bounded"


class RuntimeTerminalState(str, Enum):
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"


class RuntimeOperationOwner(str, Enum):
    PRODUCT = "product"
    RUNTIME = "runtime"


class RuntimeCapabilityDisabledReason(str, Enum):
    RUNTIME_CAPABILITY_UNSUPPORTED = "runtime_capability_unsupported"
    RUNTIME_CAPABILITY_UNAVAILABLE = "runtime_capability_unavailable"
    RUNTIME_CONFIGURATION_INVALID = "runtime_configuration_invalid"
    RUNTIME_DISABLED = "runtime_disabled"
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
    CANCELLATION_PENDING = "cancellation_pending"


def _dict(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    return dict(value or {})


def validated_disabled_operation_ids(value: Any) -> frozenset[RuntimeOperationId]:
    """Validate definition-owned disabled operations at the contract boundary."""

    if value is None:
        return frozenset()
    if not isinstance(value, (list, tuple, set, frozenset)):
        raise ValueError("runtime.features.disabled_operations must be a collection")
    raw_values = {
        item.value if isinstance(item, RuntimeOperationId) else str(item)
        for item in value
    }
    known_values = {operation.value for operation in RuntimeOperationId}
    unknown = sorted(raw_values - known_values)
    if unknown:
        raise ValueError(
            "runtime.features.disabled_operations contains unknown operations: "
            + ", ".join(unknown)
        )
    return frozenset(RuntimeOperationId(item) for item in raw_values)


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
    checkpoint_boundary_available: Optional[bool] = None

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
    modes: tuple[RuntimeCancellationMode, ...] = ()
    semantics: Optional[RuntimeCapabilitySemantics] = None
    confirmation: Optional[RuntimeConfirmationMode] = None
    terminal_states: tuple[RuntimeTerminalState, ...] = ()
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
        for values, expected, field_name in (
            (self.modes, RuntimeCancellationMode, "modes"),
            (self.terminal_states, RuntimeTerminalState, "terminal_states"),
        ):
            if not isinstance(values, tuple) or not all(isinstance(item, expected) for item in values):
                raise ValueError(f"{field_name} must contain typed enum values")
        for value, expected, field_name in (
            (self.semantics, RuntimeCapabilitySemantics, "semantics"),
            (self.confirmation, RuntimeConfirmationMode, "confirmation"),
        ):
            if value is not None and not isinstance(value, expected):
                raise ValueError(f"{field_name} must be a typed enum value or null")
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
            value["modes"] = [item.value for item in self.modes]
        if self.semantics is not None:
            value["semantics"] = self.semantics.value
        if self.confirmation is not None:
            value["confirmation"] = self.confirmation.value
        if self.terminal_states:
            value["terminal_states"] = [item.value for item in self.terminal_states]
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
    semantics: Optional[RuntimeCapabilitySemantics] = None
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
        if self.semantics is not None and not isinstance(self.semantics, RuntimeCapabilitySemantics):
            raise ValueError("semantics must be a RuntimeCapabilitySemantics value or null")

    def to_dict(self) -> Dict[str, Any]:
        value: Dict[str, Any] = {
            "support": self.support.value,
            "enabled": self.enabled,
            "disabled_reason": self.disabled_reason.value if self.disabled_reason else None,
        }
        if self.semantics is not None:
            value["semantics"] = self.semantics.value
        if self.details:
            value["details"] = dict(self.details)
        return value


@dataclass(frozen=True)
class RuntimeCapabilities:
    operations: Mapping[RuntimeOperationId, RuntimeOperationDescriptor] = field(default_factory=dict)
    features: Mapping[RuntimeFeatureId, RuntimeFeatureDescriptor] = field(default_factory=dict)
    deployment: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for operation, descriptor in self.operations.items():
            if not isinstance(operation, RuntimeOperationId):
                raise ValueError("capability operation identifiers must be RuntimeOperationId values")
            if not isinstance(descriptor, RuntimeOperationDescriptor):
                raise TypeError("capability operations must contain RuntimeOperationDescriptor values")
        for feature, descriptor in self.features.items():
            if not isinstance(feature, RuntimeFeatureId):
                raise ValueError("capability feature identifiers must be RuntimeFeatureId values")
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
                feature.value: descriptor.to_dict()
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
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        value["issues"] = [issue.to_dict() for issue in self.issues]
        return value


@dataclass(frozen=True)
class AgentRuntimeResult:
    status: str
    output: Any = None
    task_result: Optional[RuntimeTaskResult] = None
    clarification: Optional[Mapping[str, Any]] = None
    interruption: Optional[Mapping[str, Any]] = None
    artifacts: tuple[Mapping[str, Any], ...] = ()
    usage: Mapping[str, Any] = field(default_factory=dict)
    runtime_metadata: Mapping[str, Any] = field(default_factory=dict)
    continuation: Optional[ContinuationBinding] = None
    error: Optional[Mapping[str, Any]] = None
    checkpoint_boundary_available: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        value = asdict(self)
        if self.task_result is not None:
            value["task_result"] = self.task_result.to_dict()
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
class RuntimeTaskResult:
    """Usable task output without imposing a research-specific schema.

    Structured extensions are intentionally open while lifecycle vocabulary is
    closed. Runtime-native values belong in ``framework_details``.
    """

    status: RuntimeTaskResultStatus
    text: Optional[str] = None
    structured_output: Optional[Mapping[str, Any]] = None
    artifacts: tuple[RuntimeArtifact, ...] = ()
    warnings: tuple[Mapping[str, Any], ...] = ()
    gaps: tuple[str, ...] = ()
    usage: Mapping[str, Any] = field(default_factory=dict)
    error: Optional[Mapping[str, Any]] = None
    framework_details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, RuntimeTaskResultStatus):
            raise ValueError("status must be a RuntimeTaskResultStatus")
        if self.text is not None and not isinstance(self.text, str):
            raise TypeError("text must be a string or null")
        if self.text is not None and len(self.text) > 120_000:
            raise ValueError("text exceeds the runtime task result limit")
        if self.structured_output is not None and not isinstance(self.structured_output, Mapping):
            raise TypeError("structured_output must be an object or null")
        if not all(isinstance(item, RuntimeArtifact) for item in self.artifacts):
            raise TypeError("artifacts must contain RuntimeArtifact values")
        if len(self.artifacts) > 200:
            raise ValueError("artifacts exceeds the runtime task result limit")
        if not all(isinstance(item, Mapping) for item in self.warnings):
            raise TypeError("warnings must contain objects")
        if len(self.warnings) > 100:
            raise ValueError("warnings exceeds the runtime task result limit")
        if not all(isinstance(item, str) for item in self.gaps):
            raise TypeError("gaps must contain strings")
        if len(self.gaps) > 100:
            raise ValueError("gaps exceeds the runtime task result limit")
        if self.status in {
            RuntimeTaskResultStatus.COMPLETED,
            RuntimeTaskResultStatus.COMPLETED_WITH_WARNINGS,
        } and not ((self.text or "").strip() or self.structured_output or self.artifacts):
            raise ValueError("completed task results require usable output or artifacts")
        if self.status is RuntimeTaskResultStatus.COMPLETED_WITH_WARNINGS and not (self.warnings or self.gaps):
            raise ValueError("completed_with_warnings requires warnings or gaps")

    @property
    def usable(self) -> bool:
        return bool((self.text or "").strip() or self.structured_output or self.artifacts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "text": self.text,
            "structured_output": dict(self.structured_output) if self.structured_output is not None else None,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "warnings": [dict(item) for item in self.warnings],
            "gaps": list(self.gaps),
            "usage": dict(self.usage),
            "error": dict(self.error) if self.error is not None else None,
            "framework_details": dict(self.framework_details),
        }


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
    metadata: Mapping[str, Any] = field(default_factory=dict)
    context_data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "todos": [dict(value) for value in self.todos],
            "artifact_manifests": [dict(value) for value in self.artifact_manifests],
            "artifact_contents": dict(self.artifact_contents),
            "limits": dict(self.limits),
            "permissions": dict(self.permissions),
            "metadata": dict(self.metadata),
            "context_data": dict(self.context_data),
        }


def mapping_copy(value: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a shallow JSON-compatible mapping copy for adapter boundaries."""

    return _dict(value)
