"""Wire serialization and SSE helpers for the internal runtime protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from runtime_protocol import iter_sse, json_envelope, sse_encode, validate_event_mapping

from runtime_protocol.contracts import (
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    CANONICAL_RUNTIME_EVENT_KINDS,
    ContinuationBinding,
    RuntimeCapabilityDisabledReason,
    RuntimeCapabilitySemantics,
    RuntimeCancellationMode,
    RuntimeCapabilities,
    RuntimeConfirmationMode,
    RuntimeFeatureId,
    RuntimeFeatureDescriptor,
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeOperationOwner,
    RuntimePlanChange,
    RuntimeSupportLevel,
    RuntimeTerminalState,
    RuntimeTaskResultStatus,
    RuntimeTaskResult,
    RuntimeUsageSnapshot,
    RuntimeArtifact,
    RuntimeApprovalResponse,
    RuntimeCourseCorrection,
    RuntimeCourseCorrectionOutcome,
    RuntimeCourseCorrectionReceipt,
    RuntimeSteeringInput,
    RuntimeValidationIssue,
    RuntimeValidationResult,
    TaskOrchestrationDelta,
    RUNTIME_MINIMUM_COMPATIBLE_VERSION,
    RUNTIME_PROTOCOL_VERSION,
    require_protocol_fields,
)
from runtime_protocol.events import create_runtime_event


def _binding(value: Mapping[str, Any] | None) -> ContinuationBinding | None:
    if not value:
        return None
    return ContinuationBinding(
        binding_type=str(value.get("binding_type") or "unknown"),
        payload=dict(value.get("payload") or {}),
    )


def request_from_dict(value: Mapping[str, Any]) -> AgentRuntimeRequest:
    protocol_version, minimum_compatible_version = require_protocol_fields(value)
    return AgentRuntimeRequest(
        run_id=str(value["run_id"]),
        thread_id=str(value["thread_id"]),
        definition_id=str(value["definition_id"]),
        framework=str(value["framework"]),
        builder_id=str(value["builder_id"]),
        input=dict(value.get("input") or {}),
        options=dict(value.get("options") or {}),
        task_id=value.get("task_id"),
        parent_run_id=value.get("parent_run_id"),
        continuation=_binding(value.get("continuation")),
        trace_id=value.get("trace_id"),
        authentication=dict(value.get("authentication") or {}),
        permissions=dict(value.get("permissions") or {}),
        protocol_version=protocol_version,
        minimum_compatible_version=minimum_compatible_version,
    )


def course_correction_from_dict(value: Mapping[str, Any]) -> RuntimeCourseCorrection:
    protocol_version, minimum_compatible_version = require_protocol_fields(value)
    return RuntimeCourseCorrection(
        correction_id=str(value["correction_id"]),
        operation_id=str(value["operation_id"]),
        instruction=str(value["instruction"]),
        scope=str(value.get("scope") or "remaining_work"),
        observed_task_version=int(value.get("observed_task_version") or 0),
        observed_plan_revision=int(value.get("observed_plan_revision") or 0),
        submitted_at=value.get("submitted_at"),
        protocol_version=protocol_version,
        minimum_compatible_version=minimum_compatible_version,
    )


def course_correction_receipt_from_dict(
    value: Mapping[str, Any],
) -> RuntimeCourseCorrectionReceipt:
    return RuntimeCourseCorrectionReceipt(
        correction_id=str(value["correction_id"]),
        operation_id=str(value["operation_id"]),
        status=str(value["status"]),
        run_id=str(value["run_id"]),
        run_status=value.get("run_status"),
        plan_revision=(
            int(value["plan_revision"]) if value.get("plan_revision") is not None else None
        ),
    )


def course_correction_outcome_from_dict(
    value: Mapping[str, Any],
) -> RuntimeCourseCorrectionOutcome:
    return RuntimeCourseCorrectionOutcome(
        correction_id=str(value["correction_id"]),
        operation_id=str(value["operation_id"]),
        state=str(value["state"]),
        runtime_plan_revision=(
            int(value["runtime_plan_revision"])
            if value.get("runtime_plan_revision") is not None else None
        ),
        linked_run_id=str(value["linked_run_id"]) if value.get("linked_run_id") else None,
        todo_ids=tuple(str(item) for item in value.get("todo_ids") or []),
        artifact_ids=tuple(str(item) for item in value.get("artifact_ids") or []),
        explanation=str(value["explanation"]) if value.get("explanation") is not None else None,
        unresolved_reason=(
            str(value["unresolved_reason"])
            if value.get("unresolved_reason") is not None else None
        ),
    )


def definition_from_dict(value: Mapping[str, Any]) -> AgentDefinition:
    return AgentDefinition(
        definition_id=str(value["definition_id"]),
        framework=str(value["framework"]),
        builder_id=str(value["builder_id"]),
        category=value.get("category"),
        display_name=value.get("display_name"),
        capabilities=dict(value.get("capabilities") or {}),
        definition_metadata=dict(value.get("definition_metadata") or {}),
    )


def event_from_dict(value: Mapping[str, Any]) -> AgentRuntimeEvent:
    validate_event_mapping(value)
    kind = value["kind"]
    return create_runtime_event(
        event_id=str(value["event_id"]),
        run_id=str(value["run_id"]),
        sequence=int(value["sequence"]),
        kind=kind,
        attempt=int(value.get("attempt") or 1),
        payload=dict(value.get("payload") or {}),
        occurred_at=value.get("occurred_at"),
        trace_id=value.get("trace_id"),
        terminal=value.get("terminal"),
        source_metadata=dict(value.get("source_metadata") or {}),
        continuation=_binding(value.get("continuation")),
        checkpoint_boundary_available=value.get("checkpoint_boundary_available"),
    )


def result_from_dict(value: Mapping[str, Any]) -> AgentRuntimeResult:
    protocol_version, minimum_compatible_version = require_protocol_fields(value)
    task_value = value.get("task_result") if isinstance(value.get("task_result"), Mapping) else None
    task_usage = dict(task_value.get("usage") or {}) if task_value is not None else {}
    if task_usage:
        task_usage = RuntimeUsageSnapshot.from_mapping(
            task_usage,
            operation_id=str(task_usage.get("operation_id") or "runtime-operation"),
        ).to_dict()
    task_result = RuntimeTaskResult(
        status=RuntimeTaskResultStatus(str(task_value.get("status"))),
        text=task_value.get("text"),
        structured_output=dict(task_value["structured_output"]) if isinstance(task_value.get("structured_output"), Mapping) else None,
        artifacts=tuple(RuntimeArtifact(**dict(item)) for item in task_value.get("artifacts") or [] if isinstance(item, Mapping)),
        warnings=tuple(dict(item) for item in task_value.get("warnings") or [] if isinstance(item, Mapping)),
        gaps=tuple(str(item) for item in task_value.get("gaps") or []),
        usage=task_usage,
        error=dict(task_value["error"]) if isinstance(task_value.get("error"), Mapping) else None,
        framework_details=dict(task_value.get("framework_details") or {}),
        correction_outcomes=tuple(
            course_correction_outcome_from_dict(item)
            for item in task_value.get("correction_outcomes") or []
            if isinstance(item, Mapping)
        ),
    ) if task_value is not None else None
    delta_value = value.get("orchestration_delta") if isinstance(value.get("orchestration_delta"), Mapping) else None
    orchestration_delta = TaskOrchestrationDelta(
        event_id=str(delta_value["event_id"]),
        attempt_id=str(delta_value["attempt_id"]),
        operation_id=str(delta_value["operation_id"]),
        idempotency_key=str(delta_value["idempotency_key"]),
        observed_task_version=int(delta_value.get("observed_task_version") or 0),
        observed_plan_revision=int(delta_value.get("observed_plan_revision") or 0),
        plan_changes=tuple(
            RuntimePlanChange(
                runtime_revision=int(item["runtime_revision"]),
                parent_runtime_revision=int(item.get("parent_runtime_revision") or 0),
                acknowledged_product_revision=int(item.get("acknowledged_product_revision") or 0),
                reason=str(item.get("reason") or "runtime_projection"),
                planner_visit=int(item.get("planner_visit") or 1),
                plan=dict(item.get("plan") or {}),
                correction_ids=tuple(str(value) for value in item.get("correction_ids") or []),
            )
            for item in delta_value.get("plan_changes") or [] if isinstance(item, Mapping)
        ),
        todo_changes=tuple(dict(item) for item in delta_value.get("todo_changes") or [] if isinstance(item, Mapping)),
        subagent_changes=tuple(dict(item) for item in delta_value.get("subagent_changes") or [] if isinstance(item, Mapping)),
        budget_usage=dict(delta_value.get("budget_usage") or {}),
        web_access=dict(delta_value["web_access"]) if isinstance(delta_value.get("web_access"), Mapping) else None,
        artifacts=tuple(dict(item) for item in delta_value.get("artifacts") or [] if isinstance(item, Mapping)),
        pending_interrupt=dict(delta_value["pending_interrupt"]) if isinstance(delta_value.get("pending_interrupt"), Mapping) else None,
        result=dict(delta_value["result"]) if isinstance(delta_value.get("result"), Mapping) else None,
        correction_outcomes=tuple(
            course_correction_outcome_from_dict(item)
            for item in delta_value.get("correction_outcomes") or []
            if isinstance(item, Mapping)
        ),
    ) if delta_value is not None else None
    return AgentRuntimeResult(
        status=str(value.get("status") or "failed"),
        output=value.get("output"),
        task_result=task_result,
        clarification=dict(value["clarification"]) if isinstance(value.get("clarification"), Mapping) else None,
        interruption=dict(value["interruption"]) if isinstance(value.get("interruption"), Mapping) else None,
        artifacts=tuple(dict(item) for item in value.get("artifacts") or [] if isinstance(item, Mapping)),
        usage=dict(value.get("usage") or {}),
        runtime_metadata=dict(value.get("runtime_metadata") or {}),
        continuation=_binding(value.get("continuation")),
        error=dict(value["error"]) if isinstance(value.get("error"), Mapping) else None,
        checkpoint_boundary_available=value.get("checkpoint_boundary_available"),
        orchestration_delta=orchestration_delta,
        protocol_version=protocol_version,
        minimum_compatible_version=minimum_compatible_version,
    )


def validation_from_dict(value: Mapping[str, Any]) -> RuntimeValidationResult:
    protocol_version, minimum_compatible_version = require_protocol_fields(value)
    return RuntimeValidationResult(
        valid=bool(value.get("valid")),
        issues=tuple(
            RuntimeValidationIssue(
                code=str(item.get("code") or "invalid"),
                message=str(item.get("message") or "Invalid workflow"),
                path=item.get("path"),
                severity=str(item.get("severity") or "error"),
                details=dict(item.get("details") or {}),
            )
            for item in value.get("issues") or []
            if isinstance(item, Mapping)
        ),
        normalized_spec=value.get("normalized_spec"),
        runtime_metadata=dict(value.get("runtime_metadata") or {}),
        diagnostics=dict(value.get("diagnostics") or {}),
        protocol_version=protocol_version,
        minimum_compatible_version=minimum_compatible_version,
    )


def capabilities_from_dict(value: Mapping[str, Any]) -> RuntimeCapabilities:
    protocol_version, minimum_compatible_version = require_protocol_fields(value)
    if not isinstance(value, Mapping):
        raise ValueError("runtime capabilities must be an object")
    raw_operations = value.get("operations")
    if not isinstance(raw_operations, Mapping):
        raise ValueError("runtime capabilities must contain an operations object")

    operations: dict[RuntimeOperationId, RuntimeOperationDescriptor] = {}
    for operation, raw_descriptor in raw_operations.items():
        if not isinstance(operation, str) or not operation.strip():
            raise ValueError("runtime capability operation identifiers must be non-empty strings")
        if not isinstance(raw_descriptor, Mapping):
            raise ValueError(f"runtime capability descriptor for {operation!r} must be an object")
        try:
            operation_id = RuntimeOperationId(operation)
            support = RuntimeSupportLevel(raw_descriptor["support"])
            owner = RuntimeOperationOwner(raw_descriptor["owner"])
            enabled = raw_descriptor["enabled"]
            disabled_reason = raw_descriptor.get("disabled_reason")
            raw_modes = raw_descriptor.get("modes", ())
            raw_terminal_states = raw_descriptor.get("terminal_states", ())
            if not isinstance(raw_modes, (list, tuple)) or not isinstance(raw_terminal_states, (list, tuple)):
                raise ValueError("modes and terminal_states must be arrays")
            modes = tuple(RuntimeCancellationMode(item) for item in raw_modes)
            terminal_states = tuple(RuntimeTerminalState(item) for item in raw_terminal_states)
            if not isinstance(enabled, bool):
                raise ValueError("enabled must be a bool")
            if not all(isinstance(item, str) and item for item in modes + terminal_states):
                raise ValueError("modes and terminal_states must contain non-empty strings")
            disabled_reason = (
                RuntimeCapabilityDisabledReason(disabled_reason)
                if disabled_reason is not None
                else None
            )
            semantics = RuntimeCapabilitySemantics(raw_descriptor["semantics"]) if raw_descriptor.get("semantics") is not None else None
            confirmation = RuntimeConfirmationMode(raw_descriptor["confirmation"]) if raw_descriptor.get("confirmation") is not None else None
            for field_name in ("preserves_run_id", "preserves_session_id"):
                field_value = raw_descriptor.get(field_name)
                if field_value is not None and not isinstance(field_value, bool):
                    raise ValueError(f"{field_name} must be a bool or null")
            requires_runtime_binding = raw_descriptor.get("requires_runtime_binding", False)
            if not isinstance(requires_runtime_binding, bool):
                raise ValueError("requires_runtime_binding must be a bool")
            descriptor = RuntimeOperationDescriptor(
                support=support,
                owner=owner,
                enabled=enabled,
                disabled_reason=disabled_reason,
                modes=modes,
                semantics=semantics,
                confirmation=confirmation,
                terminal_states=terminal_states,
                preserves_run_id=raw_descriptor.get("preserves_run_id"),
                preserves_session_id=raw_descriptor.get("preserves_session_id"),
                requires_runtime_binding=requires_runtime_binding,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid runtime capability descriptor for {operation!r}") from exc
        operations[operation_id] = descriptor
    raw_features = value.get("features") or {}
    if not isinstance(raw_features, Mapping):
        raise ValueError("runtime capabilities features must be an object")
    features: dict[RuntimeFeatureId, RuntimeFeatureDescriptor] = {}
    for feature, raw_descriptor in raw_features.items():
        if not isinstance(feature, str) or not feature.strip() or not isinstance(raw_descriptor, Mapping):
            raise ValueError(f"invalid runtime feature descriptor for {feature!r}")
        try:
            support = RuntimeSupportLevel(raw_descriptor["support"])
            enabled = raw_descriptor["enabled"]
            disabled_reason = raw_descriptor.get("disabled_reason")
            if not isinstance(enabled, bool):
                raise ValueError("enabled must be a bool")
            disabled_reason = (
                RuntimeCapabilityDisabledReason(disabled_reason)
                if disabled_reason is not None
                else None
            )
            feature_id = RuntimeFeatureId(feature)
            semantics = RuntimeCapabilitySemantics(raw_descriptor["semantics"]) if raw_descriptor.get("semantics") is not None else None
            details = raw_descriptor.get("details") or {}
            if not isinstance(details, Mapping):
                raise ValueError("details must be an object")
            features[feature_id] = RuntimeFeatureDescriptor(
                support=support,
                enabled=enabled,
                disabled_reason=disabled_reason,
                semantics=semantics,
                details=dict(details),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid runtime feature descriptor for {feature!r}") from exc
    deployment = value.get("deployment") or {}
    if not isinstance(deployment, Mapping):
        raise ValueError("runtime capabilities deployment must be an object")
    behavior = value.get("behavior") or {}
    if not isinstance(behavior, Mapping):
        raise ValueError("runtime capabilities behavior must be an object")
    required_behavior = {
        "continuation_semantics", "usage_accounting_owner", "preserves_run_id",
        "artifact_inheritance", "supports_orchestration_delta", "required_input_fields",
    }
    if not required_behavior.issubset(behavior):
        missing = sorted(required_behavior - set(behavior))
        raise ValueError(f"runtime capabilities behavior is missing: {', '.join(missing)}")
    return RuntimeCapabilities(
        operations=operations,
        features=features,
        deployment=dict(deployment),
        behavior=dict(behavior),
        protocol_version=protocol_version,
        minimum_compatible_version=minimum_compatible_version,
    )


@dataclass(frozen=True)
class ServerEnvelope:
    status: str
    request_id: str | None = None
    result: Mapping[str, Any] | None = None
    error: Mapping[str, Any] | None = None
    runtime_metadata: Mapping[str, Any] | None = None
    protocol_version: str = RUNTIME_PROTOCOL_VERSION
    minimum_compatible_version: str = RUNTIME_MINIMUM_COMPATIBLE_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "minimum_compatible_version": self.minimum_compatible_version,
            "status": self.status,
            "request_id": self.request_id,
            "result": dict(self.result or {}),
            "error": dict(self.error or {}),
            "runtime_metadata": dict(self.runtime_metadata or {}),
        }
