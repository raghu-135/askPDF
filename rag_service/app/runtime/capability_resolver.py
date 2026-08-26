"""Effective runtime capability resolution for API consumers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from app.runtime.contracts import (
    AgentDefinition,
    RuntimeCapabilityDisabledReason,
    RuntimeCapabilities,
    RuntimeOperationId,
    RuntimeOperationDescriptor,
    RuntimeOperationOwner,
    RuntimeSupportLevel,
)
from app.runtime.adapter import AgentRuntimeAdapter
from app.runtime.errors import RuntimeError
from app.runtime.product_capabilities import product_operation_descriptors, project_public_capabilities
from app.runtime.registry import RuntimeRegistry, RuntimeSelectionError


TERMINAL_RUN_STATES = frozenset({
    "completed",
    "failed",
    "rejected",
    "expired",
    "cancelled",
    "clarification",
})
ACTIVE_RUN_OPERATIONS = frozenset({
    RuntimeOperationId.RUN_CANCEL,
    RuntimeOperationId.RUN_SEND_FOLLOWUP,
    RuntimeOperationId.RUN_INTERRUPT_WITH_INPUT,
    RuntimeOperationId.RUN_STEER_LIVE,
    RuntimeOperationId.RUN_UPDATE_STATE,
    RuntimeOperationId.RUN_CONTINUATION_CLEANUP,
    RuntimeOperationId.RUN_APPROVAL_RESPOND,
})

RESPONSE_OPERATIONS = frozenset({
    RuntimeOperationId.RUN_RESUME,
    RuntimeOperationId.RUN_APPROVAL_RESPOND,
})

TASK_ONLY_OPERATIONS = frozenset({
    RuntimeOperationId.TASK_START,
    RuntimeOperationId.TASK_PAUSE,
    RuntimeOperationId.TASK_RESUME,
    RuntimeOperationId.TASK_CANCEL,
    RuntimeOperationId.TASK_RETRY,
})

OPERATION_METHODS = {
    RuntimeOperationId.RUN_START: "start",
    RuntimeOperationId.RUN_GET: "get_run",
    RuntimeOperationId.RUN_LIST: "list_runs",
    RuntimeOperationId.RUN_WAIT: "wait",
    RuntimeOperationId.RUN_RESUME: "resume",
    RuntimeOperationId.RUN_CANCEL: "cancel",
    RuntimeOperationId.RUN_SEND_FOLLOWUP: "send_followup",
    RuntimeOperationId.RUN_INTERRUPT_WITH_INPUT: "interrupt_with_input",
    RuntimeOperationId.RUN_STEER_LIVE: "steer_live",
    RuntimeOperationId.RUN_INSPECT_STATE: "inspect_state",
    RuntimeOperationId.RUN_UPDATE_STATE: "update_state",
    RuntimeOperationId.RUN_REPLAY: "replay",
    RuntimeOperationId.RUN_FORK: "fork",
    RuntimeOperationId.RUN_APPROVAL_RESPOND: "respond_to_approval",
    RuntimeOperationId.SUBAGENT_LIST: "list_subagents",
    RuntimeOperationId.SUBAGENT_SEND: "send_to_subagent",
    RuntimeOperationId.SUBAGENT_CANCEL: "cancel_subagent",
    RuntimeOperationId.ARTIFACT_LIST: "list_artifacts",
    RuntimeOperationId.RUN_CONTINUATION_CLEANUP: "delete_continuation",
}


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
    disabled = definition.capabilities.get("disabled_operations", ())
    disabled_ids = {
        RuntimeOperationId(str(operation))
        for operation in disabled
    } if isinstance(disabled, (list, tuple, set, frozenset)) else set()
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
    """Disable declared runtime operations without a concrete adapter method."""

    operations = dict(capabilities.operations)
    for operation_id, descriptor in operations.items():
        if not descriptor.enabled or descriptor.owner is RuntimeOperationOwner.PRODUCT:
            continue
        method_name = OPERATION_METHODS.get(operation_id)
        if method_name is None:
            operations[operation_id] = _disabled(
                descriptor, RuntimeCapabilityDisabledReason.ADAPTER_OPERATION_UNMAPPED
            )
            continue
        method = getattr(adapter, method_name, None)
        declared_method = getattr(type(adapter), method_name, None)
        base_method = getattr(AgentRuntimeAdapter, method_name, None)
        if method is None or declared_method is None or declared_method is base_method:
            operations[operation_id] = _disabled(
                descriptor, RuntimeCapabilityDisabledReason.ADAPTER_OPERATION_UNIMPLEMENTED
            )
    return replace(capabilities, operations=operations)


def _with_product_operations(capabilities: RuntimeCapabilities) -> RuntimeCapabilities:
    operations = dict(capabilities.operations)
    operations.update(product_operation_descriptors())
    return replace(capabilities, operations=operations)


async def _declaration_for_adapter(adapter: Any) -> RuntimeCapabilities:
    return await adapter.deployment_capabilities()


async def capabilities_for_definition(
    definition: AgentDefinition,
    *,
    registry: RuntimeRegistry,
) -> RuntimeCapabilities:
    adapter = registry.get(definition)
    capabilities = await adapter.capabilities(definition)
    capabilities = _reconcile_implementation(capabilities, adapter)
    capabilities = _with_product_operations(capabilities)
    return apply_definition_policy(capabilities, definition)


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
    adapter = registry.get(definition)
    capabilities = await adapter.capabilities(definition)
    capabilities = _reconcile_implementation(capabilities, adapter)
    capabilities = _with_product_operations(capabilities)
    capabilities = apply_definition_policy(capabilities, definition)
    operations = dict(capabilities.operations)
    if run is None:
        for operation in TASK_ONLY_OPERATIONS - {RuntimeOperationId.TASK_START}:
            if operation in operations:
                operations[operation] = _disabled(
                    operations[operation], RuntimeCapabilityDisabledReason.TASK_RUN_NOT_CREATED
                )
        return replace(capabilities, operations=operations)

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
    if task_status not in TERMINAL_RUN_STATES and RuntimeOperationId.TASK_PAUSE in operations and task_status not in {"queued", "running"}:
        operations[RuntimeOperationId.TASK_PAUSE] = _disabled(
            operations[RuntimeOperationId.TASK_PAUSE], RuntimeCapabilityDisabledReason.TASK_NOT_PAUSEABLE
        )
    pending = getattr(run, "pending_interrupt_json", None)
    pending_type = str(pending.get("type") or "") if isinstance(pending, Mapping) else ""
    if RuntimeOperationId.TASK_RESUME in operations and task_status not in {"paused", "awaiting_human"}:
        operations[RuntimeOperationId.TASK_RESUME] = _disabled(operations[RuntimeOperationId.TASK_RESUME], RuntimeCapabilityDisabledReason.TASK_NOT_RESUMABLE)
    elif RuntimeOperationId.TASK_RESUME in operations and task_status == "awaiting_human" and pending_type != "task_pause":
        operations[RuntimeOperationId.TASK_RESUME] = _disabled(operations[RuntimeOperationId.TASK_RESUME], RuntimeCapabilityDisabledReason.TASK_NOT_RESUMABLE)
    if RuntimeOperationId.TASK_RETRY in operations and task_status not in {"failed", "expired"}:
        operations[RuntimeOperationId.TASK_RETRY] = _disabled(operations[RuntimeOperationId.TASK_RETRY], RuntimeCapabilityDisabledReason.TASK_NOT_RETRYABLE)

    if task is None and status in TERMINAL_RUN_STATES:
        for operation in TASK_ONLY_OPERATIONS:
            if operation in operations:
                operations[operation] = _disabled(operations[operation], RuntimeCapabilityDisabledReason.TASK_TERMINAL)
    elif task_status in TERMINAL_RUN_STATES:
        for operation in (
            RuntimeOperationId.TASK_START,
            RuntimeOperationId.TASK_PAUSE,
            RuntimeOperationId.TASK_RESUME,
            RuntimeOperationId.TASK_CANCEL,
        ):
            if operation in operations:
                operations[operation] = _disabled(operations[operation], RuntimeCapabilityDisabledReason.TASK_TERMINAL)

    if not binding_available and status not in TERMINAL_RUN_STATES:
        for operation, descriptor in operations.items():
            if descriptor.requires_runtime_binding:
                operations[operation] = _disabled(
                    descriptor, RuntimeCapabilityDisabledReason.RUNTIME_BINDING_UNAVAILABLE
                )

    return replace(capabilities, operations=operations)


async def require_capability(
    definition: AgentDefinition,
    operation: RuntimeOperationId | str,
    *,
    registry: RuntimeRegistry,
    run: Any | None = None,
    include_resolved_response: bool = False,
) -> RuntimeOperationDescriptor:
    """Resolve one operation and fail before an adapter call when unavailable."""

    operation_id = operation if isinstance(operation, RuntimeOperationId) else RuntimeOperationId(operation)
    capabilities = await resolve_capabilities(
        definition,
        registry=registry,
        run=run,
        include_resolved_response=include_resolved_response,
    )
    descriptor = capabilities.operations.get(operation_id)
    if descriptor is None or descriptor.support is RuntimeSupportLevel.UNSUPPORTED:
        raise RuntimeError.capability_unsupported(
            operation_id=operation_id.value,
            framework=definition.framework,
            builder_id=definition.builder_id,
            explanation="The runtime does not provide this operation",
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
    value: dict[str, Any] = {
        "resource": resource,
        "runtime_id": runtime_id,
        "framework": framework,
        "builder_id": builder_id,
        "available": capabilities is not None,
        "capabilities": project_public_capabilities(capabilities).to_dict() if capabilities is not None else None,
    }
    if definition_id is not None:
        value["definition_id"] = definition_id
    if run_id is not None:
        value["run_id"] = run_id
    if run_status is not None:
        value["run_status"] = run_status
    if error is not None:
        value["error"] = dict(error)
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
    try:
        capabilities = await _declaration_for_adapter(adapter)
        capabilities = _reconcile_implementation(capabilities, adapter)
        return _with_product_operations(capabilities), None
    except RuntimeError as exc:
        return None, capability_discovery_error(exc, adapter)
    except Exception as exc:
        return None, capability_discovery_error(exc, adapter)
