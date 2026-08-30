"""Stable, framework-neutral contracts for agent execution.

Only Python's standard library is allowed in this package. Contract values are
JSON-compatible and shared by the control plane and every runtime service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional


CONTRACT_VERSION = 1
RUNTIME_OPERATION_EVENT_KINDS = frozenset({
    "operation.started",
    "operation.completed",
    "operation.failed",
    "operation.skipped",
})


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
class RuntimeCapabilities:
    streaming: bool = False
    resume: bool = False
    cancellation: bool = False
    inspection: bool = False
    continuation_cleanup: bool = False
    task_execution: bool = False
    native_checkpoints: bool = False
    runtime_version: Optional[str] = None
    contract_version: int = CONTRACT_VERSION

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
