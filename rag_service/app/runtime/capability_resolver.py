"""Effective runtime capability resolution for API consumers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Mapping

from app.runtime.contracts import (
    AgentDefinition,
    RuntimeCapabilityDisabledReason,
    RuntimeCapabilities,
    RuntimeOperationId,
    RuntimeOperationDescriptor,
    RuntimeOperationOwner,
    RuntimeSupportLevel,
    validated_disabled_operation_ids,
)
from app.runtime.adapter import AgentRuntimeAdapter
from app.runtime.errors import RuntimeError
from app.runtime.product_capabilities import product_operation_descriptors, project_public_capabilities
from app.runtime.registry import RuntimeRegistry, RuntimeSelectionError
from app.db.enums import AgentRunStatus
from app.models.deep_research import AgentTaskStatus


logger = logging.getLogger(__name__)


TERMINAL_RUN_STATES = frozenset({
    status.value for status in (
        AgentRunStatus.COMPLETED,
        AgentRunStatus.FAILED,
        AgentRunStatus.REJECTED,
        AgentRunStatus.EXPIRED,
        AgentRunStatus.CANCELLED,
        AgentRunStatus.CLARIFICATION,
    )
})
TERMINAL_TASK_STATES = frozenset({
    AgentTaskStatus.COMPLETED.value,
    AgentTaskStatus.FAILED.value,
    AgentTaskStatus.EXPIRED.value,
    AgentTaskStatus.CANCELLED.value,
})
PAUSEABLE_TASK_STATES = frozenset({
    AgentTaskStatus.QUEUED.value,
    AgentTaskStatus.RUNNING.value,
})
RESUMABLE_TASK_STATES = frozenset({
    AgentTaskStatus.PAUSED.value,
    AgentTaskStatus.AWAITING_APPROVAL.value,
})
RETRYABLE_TASK_STATES = frozenset({
    AgentTaskStatus.FAILED.value,
    AgentTaskStatus.EXPIRED.value,
})
ACTIVE_RUN_OPERATIONS = frozenset({
    RuntimeOperationId.RUN_CANCEL,
    RuntimeOperationId.RUN_SEND_FOLLOWUP,
    RuntimeOperationId.RUN_INTERRUPT_WITH_INPUT,
    RuntimeOperationId.RUN_STEER_LIVE,
    RuntimeOperationId.RUN_CONTINUATION_CLEANUP,
    RuntimeOperationId.RUN_APPROVAL_RESPOND,
})


@dataclass(frozen=True)
class CapabilityResolution:
    """One authoritative capability result for all product API layers."""

    capabilities: RuntimeCapabilities
    error: Mapping[str, Any] | None = None
    runtime_available: bool = True


RESPONSE_OPERATIONS = frozenset({
    RuntimeOperationId.RUN_RESUME,
    RuntimeOperationId.RUN_APPROVAL_RESPOND,
    RuntimeOperationId.TASK_RESULT_REVIEW_RESPOND,
    RuntimeOperationId.TASK_BUDGET_REVIEW_RESPOND,
})

TASK_ONLY_OPERATIONS = frozenset({
    RuntimeOperationId.TASK_START,
    RuntimeOperationId.TASK_PAUSE,
    RuntimeOperationId.TASK_RESUME,
    RuntimeOperationId.TASK_CANCEL,
    RuntimeOperationId.TASK_RETRY,
    RuntimeOperationId.TASK_RESULT_REVIEW_RESPOND,
    RuntimeOperationId.TASK_BUDGET_REVIEW_RESPOND,
    RuntimeOperationId.TASK_COURSE_CORRECTION_SUBMIT,
})

CHECKPOINT_OPERATIONS = frozenset({
    RuntimeOperationId.RUN_RESUME,
    RuntimeOperationId.RUN_INSPECT_STATE,
    RuntimeOperationId.RUN_REPLAY,
    RuntimeOperationId.RUN_FORK,
    RuntimeOperationId.RUN_CONTINUATION_CLEANUP,
})

def deployment_id(adapter: Any) -> str:
    return f"{adapter.framework}:{adapter.builder_id}"


def _disabled(
    descriptor: RuntimeOperationDescriptor,
    reason: RuntimeCapabilityDisabledReason,
) -> RuntimeOperationDescriptor:
    if descriptor.support is RuntimeSupportLevel.UNSUPPORTED or not descriptor.enabled:
        return descriptor
    return replace(descriptor, enabled=False, disabled_reason=reason)


def apply_definition_policy(
    capabilities: RuntimeCapabilities,
    definition: AgentDefinition,
) -> RuntimeCapabilities:
    disabled_ids = validated_disabled_operation_ids(
        definition.capabilities.get("disabled_operations")
    )
    operations = {
        operation: _disabled(descriptor, RuntimeCapabilityDisabledReason.DEFINITION_POLICY)
        if operation in disabled_ids
        else descriptor
        for operation, descriptor in capabilities.operations.items()
    }
    if bool(definition.capabilities.get("supports_long_running_tasks")):
        for operation in TASK_ONLY_OPERATIONS:
            descriptor = operations.get(operation)
            if descriptor is not None and descriptor.disabled_reason is RuntimeCapabilityDisabledReason.DEFINITION_NOT_TASK_RUNTIME:
                operations[operation] = replace(descriptor, enabled=True, disabled_reason=None)
    else:
        for operation in TASK_ONLY_OPERATIONS:
            if operation in operations:
                operations[operation] = _disabled(
                    operations[operation], RuntimeCapabilityDisabledReason.DEFINITION_NOT_TASK_RUNTIME
                )
    return replace(capabilities, operations=operations)


def _reconcile_implementation(
    capabilities: RuntimeCapabilities,
    adapter: Any,
) -> RuntimeCapabilities:
    """Disable declarations not explicitly registered by the adapter."""

    operations = dict(capabilities.operations)
    registered = frozenset(getattr(adapter, "implemented_operations", frozenset()))
    for operation_id, descriptor in operations.items():
        if not descriptor.enabled or descriptor.owner is RuntimeOperationOwner.PRODUCT:
            continue
        if operation_id not in registered:
            operations[operation_id] = _disabled(
                descriptor, RuntimeCapabilityDisabledReason.ADAPTER_OPERATION_UNIMPLEMENTED
            )
    return replace(capabilities, operations=operations)


def _with_product_operations(capabilities: RuntimeCapabilities) -> RuntimeCapabilities:
    operations = dict(capabilities.operations)
    operations.update(product_operation_descriptors())
    return replace(capabilities, operations=operations)


def _failure_disabled_reason(error: Mapping[str, Any]) -> RuntimeCapabilityDisabledReason:
    code = str(error.get("code") or "")
    if code == RuntimeCapabilityDisabledReason.RUNTIME_CONFIGURATION_INVALID.value:
        return RuntimeCapabilityDisabledReason.RUNTIME_CONFIGURATION_INVALID
    if code == RuntimeCapabilityDisabledReason.RUNTIME_DISABLED.value:
        return RuntimeCapabilityDisabledReason.RUNTIME_DISABLED
    return RuntimeCapabilityDisabledReason.RUNTIME_UNAVAILABLE


def _unavailable_product_capabilities(adapter: Any, error: Mapping[str, Any]) -> RuntimeCapabilities:
    disabled_reason = _failure_disabled_reason(error)
    operations = product_operation_descriptors()
    for operation in TASK_ONLY_OPERATIONS | {RuntimeOperationId.ARTIFACT_LIST}:
        descriptor = operations.get(operation)
        if descriptor is not None:
            operations[operation] = _disabled(
                descriptor, disabled_reason
            )
    return RuntimeCapabilities(
        operations=operations,
        deployment={
            "framework": adapter.framework,
            "builder_id": adapter.builder_id,
            "runtime_available": False,
            "discovery_error": error.get("code") or "runtime_unavailable",
            "discovery_safe_message": error.get("safe_message") or error.get("message"),
            "discovery_retryable": bool(error.get("retryable")),
            "discovery_details": dict(error.get("details") or {}),
        },
    )


def checkpoint_boundary_available(run: Any) -> bool:
    """Read the persisted explicit run fact; a binding is not a boundary."""

    metadata = getattr(run, "run_metadata_json", None)
    return isinstance(metadata, Mapping) and metadata.get("checkpoint_boundary_available") is True


def _apply_task_cancel_dependency(
    operations: dict[RuntimeOperationId, RuntimeOperationDescriptor],
    *,
    submitted: bool,
) -> None:
    """A submitted task can cancel only when its effective runtime run can."""

    task_cancel = operations.get(RuntimeOperationId.TASK_CANCEL)
    if task_cancel is None or not submitted:
        return
    run_cancel = operations.get(RuntimeOperationId.RUN_CANCEL)
    if run_cancel is None:
        operations[RuntimeOperationId.TASK_CANCEL] = replace(
            task_cancel,
            enabled=False,
            disabled_reason=RuntimeCapabilityDisabledReason.RUNTIME_CAPABILITY_UNSUPPORTED,
        )
        return
    if not run_cancel.enabled or run_cancel.support is RuntimeSupportLevel.UNSUPPORTED:
        operations[RuntimeOperationId.TASK_CANCEL] = replace(
            task_cancel,
            enabled=False,
            disabled_reason=run_cancel.disabled_reason
            or RuntimeCapabilityDisabledReason.RUNTIME_CAPABILITY_UNSUPPORTED,
        )


def _apply_task_start_dependency(
    operations: dict[RuntimeOperationId, RuntimeOperationDescriptor],
) -> None:
    """A task can start only when its runtime run can start."""

    task_start = operations.get(RuntimeOperationId.TASK_START)
    if task_start is None or not task_start.enabled:
        return
    run_start = operations.get(RuntimeOperationId.RUN_START)
    if run_start is None or not run_start.enabled or run_start.support is RuntimeSupportLevel.UNSUPPORTED:
        operations[RuntimeOperationId.TASK_START] = replace(
            task_start,
            enabled=False,
            disabled_reason=(
                run_start.disabled_reason
                if run_start is not None and run_start.disabled_reason is not None
                else RuntimeCapabilityDisabledReason.RUNTIME_CAPABILITY_UNSUPPORTED
            ),
        )


async def _reconciled_capabilities(
    adapter: Any,
    definition: AgentDefinition | None = None,
) -> RuntimeCapabilities:
    """Apply the common capability pipeline for every resolution level."""

    capabilities = await (
        adapter.capabilities(definition)
        if definition is not None
        else adapter.deployment_capabilities()
    )
    capabilities = _reconcile_implementation(capabilities, adapter)
    capabilities = _with_product_operations(capabilities)
    if definition is not None:
        capabilities = apply_definition_policy(capabilities, definition)
    operations = dict(capabilities.operations)
    _apply_task_start_dependency(operations)
    return replace(capabilities, operations=operations)


async def resolve_definition_capability_resolution(
    definition: AgentDefinition,
    *,
    registry: RuntimeRegistry,
    adapter: AgentRuntimeAdapter | None = None,
) -> CapabilityResolution:
    adapter = adapter or registry.get(definition)
    try:
        capabilities = await _reconciled_capabilities(adapter, definition)
    except RuntimeError as exc:
        error = capability_discovery_error(exc, adapter)
        return CapabilityResolution(_unavailable_product_capabilities(adapter, error), error, False)
    artifact_list = capabilities.operations.get(RuntimeOperationId.ARTIFACT_LIST)
    if artifact_list is not None:
        capabilities = replace(
            capabilities,
            operations={
                **capabilities.operations,
                RuntimeOperationId.ARTIFACT_LIST: _disabled(
                    artifact_list, RuntimeCapabilityDisabledReason.TASK_RUN_NOT_CREATED
                ),
            },
        )
    return CapabilityResolution(
        capabilities,
        _deployment_error(capabilities),
        capabilities.deployment.get("runtime_available", True) is not False,
    )


async def capabilities_for_definition(
    definition: AgentDefinition,
    *,
    registry: RuntimeRegistry,
) -> RuntimeCapabilities:
    return (await resolve_definition_capability_resolution(definition, registry=registry)).capabilities


async def resolve_deployment_capability_resolution(
    adapter: Any,
) -> CapabilityResolution:
    """Resolve one runtime deployment without applying definition/run policy."""

    try:
        capabilities = await _reconciled_capabilities(adapter)
    except RuntimeError as exc:
        error = capability_discovery_error(exc, adapter)
        return CapabilityResolution(
            _unavailable_product_capabilities(adapter, error),
            error,
            runtime_available=False,
        )
    except Exception:
        logger.exception(
            "Unexpected capability resolution failure | framework=%s builder_id=%s",
            adapter.framework,
            adapter.builder_id,
        )
        raise

    artifact_list = capabilities.operations.get(RuntimeOperationId.ARTIFACT_LIST)
    if artifact_list is not None:
        capabilities = replace(
            capabilities,
            operations={
                **capabilities.operations,
                RuntimeOperationId.ARTIFACT_LIST: _disabled(
                    artifact_list, RuntimeCapabilityDisabledReason.TASK_RUN_NOT_CREATED
                ),
            },
        )
    error = _deployment_error(capabilities)
    return CapabilityResolution(
        capabilities,
        error,
        runtime_available=capabilities.deployment.get("runtime_available", True) is not False,
    )


def pending_interrupt_response_operation(
    run: Any,
    *,
    include_resolved: bool = False,
) -> RuntimeOperationId | None:
    """Return the explicitly declared response operation for a pending run interrupt."""

    run_status = str(getattr(run, "status", "") or "")
    if run_status != "awaiting_human" and not include_resolved:
        return None
    payload = getattr(run, "pending_interrupt_json", None)
    if not isinstance(payload, Mapping):
        return None
    pending_status = str(payload.get("status") or "")
    if pending_status != "pending" and not (include_resolved and pending_status in {"resumed", "resolved"}):
        return None
    value = payload.get("response_operation")
    if not isinstance(value, str):
        return None
    try:
        operation = RuntimeOperationId(value)
    except ValueError:
        return None
    return operation if operation in RESPONSE_OPERATIONS else None


async def resolve_capabilities(
    definition: AgentDefinition,
    *,
    registry: RuntimeRegistry,
    run: Any | None = None,
    task: Any | None = None,
    include_resolved_response: bool = False,
) -> RuntimeCapabilities:
    resolution = await (
        resolve_run_capability_resolution(definition, registry=registry, run=run, task=task,
                                          include_resolved_response=include_resolved_response)
        if run is not None
        else resolve_definition_capability_resolution(definition, registry=registry)
    )
    return resolution.capabilities


async def resolve_run_capability_resolution(
    definition: AgentDefinition,
    *,
    registry: RuntimeRegistry,
    run: Any,
    task: Any | None = None,
    include_resolved_response: bool = False,
    adapter: AgentRuntimeAdapter | None = None,
) -> CapabilityResolution:
    adapter = adapter or registry.get(definition)
    try:
        capabilities = await _reconciled_capabilities(adapter, definition)
    except RuntimeError as exc:
        error = capability_discovery_error(exc, adapter)
        capabilities = _unavailable_product_capabilities(adapter, error)
        return CapabilityResolution(capabilities, error, runtime_available=False)
    except Exception:
        logger.exception(
            "Unexpected capability resolution failure | framework=%s builder_id=%s definition_id=%s run_id=%s",
            definition.framework,
            definition.builder_id,
            definition.definition_id,
            getattr(run, "id", None),
        )
        raise
    operations = dict(capabilities.operations)

    if task is None:
        artifact_list = operations.get(RuntimeOperationId.ARTIFACT_LIST)
        if artifact_list is not None:
            operations[RuntimeOperationId.ARTIFACT_LIST] = _disabled(
                artifact_list, RuntimeCapabilityDisabledReason.TASK_RUN_NOT_CREATED
            )

    if RuntimeOperationId.RUN_START in operations and not getattr(run, "_fresh_runtime_run", False):
        operations[RuntimeOperationId.RUN_START] = _disabled(
            operations[RuntimeOperationId.RUN_START], RuntimeCapabilityDisabledReason.RUN_ALREADY_CREATED
        )
    status = str(getattr(run, "status", "") or "")
    if RuntimeOperationId.TASK_START in operations:
        operations[RuntimeOperationId.TASK_START] = _disabled(
            operations[RuntimeOperationId.TASK_START], RuntimeCapabilityDisabledReason.TASK_ALREADY_STARTED
        )
    pending_operation = pending_interrupt_response_operation(run, include_resolved=include_resolved_response)
    binding = getattr(run, "runtime_binding_json", None)
    binding_available = bool(binding) and str(getattr(run, "runtime_binding_status", "active")) == "active"
    run_metadata = getattr(run, "run_metadata_json", None)
    submitted = isinstance(run_metadata, Mapping) and run_metadata.get("runtime_started") is True

    if status in TERMINAL_RUN_STATES:
        for operation in ACTIVE_RUN_OPERATIONS | RESPONSE_OPERATIONS:
            if operation in operations:
                operations[operation] = _disabled(operations[operation], RuntimeCapabilityDisabledReason.RUN_TERMINAL)
    else:
        for operation in RESPONSE_OPERATIONS:
            if operation in operations and operation != pending_operation:
                operations[operation] = _disabled(
                    operations[operation], RuntimeCapabilityDisabledReason.NO_PENDING_INTERRUPT
                )

    task_status = str(getattr(task, "status", "") or status)
    budget_review = operations.get(RuntimeOperationId.TASK_BUDGET_REVIEW_RESPOND)
    if budget_review is not None:
        operations[RuntimeOperationId.TASK_BUDGET_REVIEW_RESPOND] = replace(
            budget_review,
            preserves_run_id=definition.framework == "langgraph",
            preserves_session_id=True,
        )
    course_correction = operations.get(RuntimeOperationId.TASK_COURSE_CORRECTION_SUBMIT)
    if course_correction is not None and task_status not in {
        AgentTaskStatus.RUNNING.value, AgentTaskStatus.QUEUED.value,
    }:
        operations[RuntimeOperationId.TASK_COURSE_CORRECTION_SUBMIT] = _disabled(
            course_correction, RuntimeCapabilityDisabledReason.TASK_TERMINAL
            if task_status in TERMINAL_TASK_STATES else RuntimeCapabilityDisabledReason.TASK_NOT_PAUSEABLE,
        )
    run_metadata = getattr(run, "run_metadata_json", None)
    cancellation_pending = (
        status not in TERMINAL_RUN_STATES
        and (
            task_status == AgentTaskStatus.CANCELLING.value
            or isinstance(run_metadata, Mapping)
            and run_metadata.get("cancel_requested") is True
        )
    )
    if cancellation_pending:
        for operation in (RuntimeOperationId.RUN_CANCEL, RuntimeOperationId.TASK_CANCEL):
            descriptor = operations.get(operation)
            if descriptor is not None:
                operations[operation] = _disabled(
                    descriptor, RuntimeCapabilityDisabledReason.CANCELLATION_PENDING
                )
    if task_status not in TERMINAL_TASK_STATES and RuntimeOperationId.TASK_PAUSE in operations and task_status not in PAUSEABLE_TASK_STATES:
        operations[RuntimeOperationId.TASK_PAUSE] = _disabled(
            operations[RuntimeOperationId.TASK_PAUSE], RuntimeCapabilityDisabledReason.TASK_NOT_PAUSEABLE
        )
    pending = getattr(run, "pending_interrupt_json", None)
    pending_type = str(pending.get("type") or "") if isinstance(pending, Mapping) else ""
    if RuntimeOperationId.TASK_RESUME in operations and task_status not in RESUMABLE_TASK_STATES:
        operations[RuntimeOperationId.TASK_RESUME] = _disabled(operations[RuntimeOperationId.TASK_RESUME], RuntimeCapabilityDisabledReason.TASK_NOT_RESUMABLE)
    elif RuntimeOperationId.TASK_RESUME in operations and task_status == AgentTaskStatus.AWAITING_APPROVAL.value and pending_type != "task_pause":
        operations[RuntimeOperationId.TASK_RESUME] = _disabled(operations[RuntimeOperationId.TASK_RESUME], RuntimeCapabilityDisabledReason.TASK_NOT_RESUMABLE)
    if RuntimeOperationId.TASK_RETRY in operations and task_status not in RETRYABLE_TASK_STATES:
        operations[RuntimeOperationId.TASK_RETRY] = _disabled(operations[RuntimeOperationId.TASK_RETRY], RuntimeCapabilityDisabledReason.TASK_NOT_RETRYABLE)

    if task is None and status in TERMINAL_RUN_STATES:
        for operation in TASK_ONLY_OPERATIONS:
            if operation in operations:
                operations[operation] = _disabled(operations[operation], RuntimeCapabilityDisabledReason.TASK_TERMINAL)
    elif task_status in TERMINAL_TASK_STATES:
        for operation in (
            RuntimeOperationId.TASK_START,
            RuntimeOperationId.TASK_PAUSE,
            RuntimeOperationId.TASK_RESUME,
            RuntimeOperationId.TASK_CANCEL,
        ):
            if operation in operations:
                operations[operation] = _disabled(operations[operation], RuntimeCapabilityDisabledReason.TASK_TERMINAL)

    _apply_task_cancel_dependency(operations, submitted=submitted)
    _apply_task_start_dependency(operations)

    if not binding_available and status not in TERMINAL_RUN_STATES:
        for operation, descriptor in operations.items():
            if operation in CHECKPOINT_OPERATIONS or descriptor.requires_runtime_binding:
                operations[operation] = _disabled(
                    descriptor, RuntimeCapabilityDisabledReason.RUNTIME_BINDING_UNAVAILABLE
                )

    if not checkpoint_boundary_available(run):
        for operation in CHECKPOINT_OPERATIONS:
            descriptor = operations.get(operation)
            if descriptor is not None and descriptor.enabled:
                operations[operation] = _disabled(
                    descriptor, RuntimeCapabilityDisabledReason.RUN_NOT_CHECKPOINT_BOUNDARY
                )

    capabilities = replace(capabilities, operations=operations)
    return CapabilityResolution(
        capabilities,
        _deployment_error(capabilities),
        runtime_available=capabilities.deployment.get("runtime_available", True) is not False,
    )


def _deployment_error(capabilities: RuntimeCapabilities) -> dict[str, Any] | None:
    if capabilities.deployment.get("runtime_available") is not False:
        return None
    deployment = capabilities.deployment
    configuration_error = deployment.get("configuration_error")
    code = str(
        deployment.get("discovery_error")
        or (RuntimeCapabilityDisabledReason.RUNTIME_CONFIGURATION_INVALID.value if configuration_error else "runtime_unavailable")
    )
    return {
        "code": code,
        "safe_message": str(deployment.get("discovery_safe_message") or configuration_error or "Agent runtime deployment is unavailable"),
        "retryable": bool(deployment.get("discovery_retryable", False if configuration_error else True)),
        "details": dict(deployment.get("discovery_details") or deployment),
    }


async def require_capability(
    definition: AgentDefinition,
    operation: RuntimeOperationId | str,
    *,
    registry: RuntimeRegistry,
    run: Any | None = None,
    task: Any | None = None,
    include_resolved_response: bool = False,
) -> RuntimeOperationDescriptor:
    """Resolve one operation and fail before an adapter call when unavailable."""

    operation_id = operation if isinstance(operation, RuntimeOperationId) else RuntimeOperationId(operation)
    if run is None:
        resolution = await resolve_definition_capability_resolution(
            definition,
            registry=registry,
        )
    else:
        resolution = await resolve_run_capability_resolution(
            definition,
            registry=registry,
            run=run,
            task=task,
            include_resolved_response=include_resolved_response,
        )
    capabilities = resolution.capabilities
    descriptor = capabilities.operations.get(operation_id)
    if not resolution.runtime_available:
        error = dict(resolution.error or {})
        disabled_reason = str(
            error.get("code")
            or RuntimeCapabilityDisabledReason.RUNTIME_UNAVAILABLE.value
        )
        raise RuntimeError.capability_unavailable(
            operation_id=operation_id.value,
            framework=definition.framework,
            builder_id=definition.builder_id,
            support_level=(descriptor.support.value if descriptor is not None else RuntimeSupportLevel.CONDITIONAL.value),
            disabled_reason=disabled_reason,
            retryable=bool(error.get("retryable", False)),
            code=str(error.get("code") or "runtime_capability_unavailable"),
            safe_message=str(error.get("safe_message") or "The runtime deployment is unavailable"),
            details=dict(error.get("details") or {}),
        )
    if descriptor is None or descriptor.support is RuntimeSupportLevel.UNSUPPORTED:
        raise RuntimeError.capability_unsupported(
            operation_id=operation_id.value,
            framework=definition.framework,
            builder_id=definition.builder_id,
            explanation="The runtime does not provide this operation",
        )
    if run is None and operation_id in TASK_ONLY_OPERATIONS - {RuntimeOperationId.TASK_START}:
        raise RuntimeError.capability_unavailable(
            operation_id=operation_id.value,
            framework=definition.framework,
            builder_id=definition.builder_id,
            support_level=descriptor.support.value,
            disabled_reason=RuntimeCapabilityDisabledReason.TASK_RUN_NOT_CREATED.value,
            retryable=False,
        )
    if not descriptor.enabled:
        raise RuntimeError.capability_unavailable(
            operation_id=operation_id.value,
            framework=definition.framework,
            builder_id=definition.builder_id,
            support_level=descriptor.support.value,
            disabled_reason=(
                descriptor.disabled_reason or RuntimeCapabilityDisabledReason.RUNTIME_CAPABILITY_UNAVAILABLE
            ).value,
            retryable=descriptor.disabled_reason in {
                RuntimeCapabilityDisabledReason.RUNTIME_UNAVAILABLE,
            },
        )
    return descriptor


def capability_envelope(
    *,
    capabilities: RuntimeCapabilities | None,
    resource: str,
    runtime_id: str,
    framework: str,
    builder_id: str,
    definition_id: str | None = None,
    run_id: str | None = None,
    run_status: str | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    effective_error = error or (_deployment_error(capabilities) if capabilities is not None else None)
    value: dict[str, Any] = {
        "resource": resource,
        "runtime_id": runtime_id,
        "framework": framework,
        "builder_id": builder_id,
        "runtime_available": (
            capabilities is not None
            and capabilities.deployment.get("runtime_available", True) is not False
        ),
        "capabilities": project_public_capabilities(capabilities).to_dict() if capabilities is not None else None,
    }
    if definition_id is not None:
        value["definition_id"] = definition_id
    if run_id is not None:
        value["run_id"] = run_id
    if run_status is not None:
        value["run_status"] = run_status
    if effective_error is not None:
        value["error"] = dict(effective_error)
    return value


def capability_discovery_error(exc: BaseException, adapter: Any) -> dict[str, Any]:
    if isinstance(exc, RuntimeError):
        return exc.to_dict()
    return RuntimeError.from_exception(
        exc,
        code="runtime_capability_discovery_failed",
        safe_message="Runtime capability discovery failed",
        details={"framework": adapter.framework, "builder_id": adapter.builder_id},
    ).to_dict()


async def discover_adapter_capabilities(
    adapter: Any,
) -> tuple[RuntimeCapabilities | None, dict[str, Any] | None]:
    resolution = await resolve_deployment_capability_resolution(adapter)
    return resolution.capabilities, resolution.error
