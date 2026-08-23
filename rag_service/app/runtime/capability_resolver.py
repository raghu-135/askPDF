"""Effective runtime capability resolution for API consumers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from app.runtime.contracts import (
    AgentDefinition,
    RuntimeCapabilities,
    RuntimeOperationId,
    RuntimeOperationDescriptor,
    RuntimeOperationOwner,
    RuntimeSupportLevel,
)
from app.runtime.adapter import AgentRuntimeAdapter
from app.runtime.errors import RuntimeError
from app.runtime.registry import RuntimeRegistry, RuntimeSelectionError


TERMINAL_RUN_STATES = frozenset({
    "completed",
    "failed",
    "rejected",
    "expired",
    "cancelled",
})
ACTIVE_RUN_OPERATIONS = frozenset({
    RuntimeOperationId.RUN_CANCEL.value,
    RuntimeOperationId.RUN_PAUSE.value,
    RuntimeOperationId.RUN_RESUME.value,
    RuntimeOperationId.RUN_SEND_FOLLOWUP.value,
    RuntimeOperationId.RUN_INTERRUPT_WITH_INPUT.value,
    RuntimeOperationId.RUN_STEER_LIVE.value,
    RuntimeOperationId.RUN_UPDATE_STATE.value,
    RuntimeOperationId.RUN_CONTINUE.value,
    RuntimeOperationId.RUN_CONTINUATION_CLEANUP.value,
    RuntimeOperationId.RUN_APPROVAL_RESPOND.value,
    RuntimeOperationId.INTERRUPT_RESPOND.value,
})

RESPONSE_OPERATIONS = frozenset({
    RuntimeOperationId.RUN_RESUME.value,
    RuntimeOperationId.RUN_APPROVAL_RESPOND.value,
    RuntimeOperationId.INTERRUPT_RESPOND.value,
})

TASK_ONLY_OPERATIONS = frozenset({
    RuntimeOperationId.RUN_PAUSE.value,
    RuntimeOperationId.RUN_RETRY.value,
})

OPERATION_METHODS = {
    "run.start": "start",
    "run.get": "get_run",
    "run.list": "list_runs",
    "run.wait": "wait",
    "run.events": "stream_events",
    "run.resume": "resume",
    "run.cancel": "cancel",
    "run.send_followup": "send_followup",
    "run.interrupt_with_input": "interrupt_with_input",
    "run.steer_live": "steer_live",
    "run.inspect_state": "inspect_state",
    "run.update_state": "update_state",
    "run.replay": "replay",
    "run.fork": "fork",
    "run.approval.respond": "respond_to_approval",
    "interrupt.respond": "respond_to_interrupt",
    "subagent.list": "list_subagents",
    "subagent.send": "send_to_subagent",
    "subagent.cancel": "cancel_subagent",
    "artifact.list": "list_artifacts",
    "run.continuation.cleanup": "delete_continuation",
}


def deployment_id(adapter: Any) -> str:
    return f"{adapter.framework}:{adapter.builder_id}"


def _disabled(
    descriptor: RuntimeOperationDescriptor,
    reason: str,
) -> RuntimeOperationDescriptor:
    if descriptor.support is RuntimeSupportLevel.UNSUPPORTED or not descriptor.enabled:
        return descriptor
    return replace(descriptor, enabled=False, disabled_reason=reason)


def apply_definition_policy(
    capabilities: RuntimeCapabilities,
    definition: AgentDefinition,
) -> RuntimeCapabilities:
    disabled = definition.capabilities.get("disabled_operations", ())
    if not isinstance(disabled, (list, tuple, set, frozenset)):
        return capabilities
    disabled_ids = {str(operation) for operation in disabled}
    operations = {
        operation: _disabled(descriptor, "definition_policy")
        if (operation.value if isinstance(operation, RuntimeOperationId) else str(operation)) in disabled_ids
        else descriptor
        for operation, descriptor in capabilities.operations.items()
    }
    return replace(capabilities, operations=operations)


async def capabilities_for_definition(
    definition: AgentDefinition,
    *,
    registry: RuntimeRegistry,
) -> RuntimeCapabilities:
    adapter = registry.get(definition)
    capabilities = await adapter.capabilities(definition)
    return apply_definition_policy(capabilities, definition)


def pending_interrupt_response_operation(run: Any) -> RuntimeOperationId | None:
    """Return the explicitly declared response operation for a pending run interrupt."""

    if str(getattr(run, "status", "") or "") != "awaiting_human":
        return None
    payload = getattr(run, "pending_interrupt_json", None)
    if not isinstance(payload, Mapping) or payload.get("status") != "pending":
        return None
    value = payload.get("response_operation")
    if not isinstance(value, str) or value not in RESPONSE_OPERATIONS:
        return None
    return RuntimeOperationId(value)


async def resolve_capabilities(
    definition: AgentDefinition,
    *,
    registry: RuntimeRegistry,
    run: Any | None = None,
) -> RuntimeCapabilities:
    capabilities = await capabilities_for_definition(definition, registry=registry)
    if run is None:
        return capabilities

    operations = dict(capabilities.operations)
    supports_long_running_tasks = bool(definition.capabilities.get("supports_long_running_tasks"))
    if not supports_long_running_tasks:
        for operation in TASK_ONLY_OPERATIONS:
            if operation in operations:
                operations[operation] = _disabled(operations[operation], "definition_not_task_runtime")
    status = str(getattr(run, "status", "") or "")
    pending_operation = pending_interrupt_response_operation(run)
    binding = getattr(run, "runtime_binding_json", None)
    binding_available = bool(binding) and str(getattr(run, "runtime_binding_status", "active")) == "active"

    if status in TERMINAL_RUN_STATES:
        for operation in ACTIVE_RUN_OPERATIONS:
            if operation in operations:
                operations[operation] = _disabled(operations[operation], "run_terminal")
    else:
        for operation in RESPONSE_OPERATIONS:
            if operation in operations and operation != (pending_operation.value if pending_operation else None):
                operations[operation] = _disabled(operations[operation], "no_pending_interrupt")

    if status not in TERMINAL_RUN_STATES and RuntimeOperationId.RUN_PAUSE.value in operations and status not in {"queued", "running"}:
        operations[RuntimeOperationId.RUN_PAUSE.value] = _disabled(operations[RuntimeOperationId.RUN_PAUSE.value], "run_not_pauseable")
    if RuntimeOperationId.RUN_RETRY.value in operations and status not in {"failed", "expired"}:
        operations[RuntimeOperationId.RUN_RETRY.value] = _disabled(operations[RuntimeOperationId.RUN_RETRY.value], "run_not_retryable")

    if not binding_available and status not in TERMINAL_RUN_STATES:
        for operation, descriptor in operations.items():
            if descriptor.requires_runtime_binding:
                operations[operation] = _disabled(descriptor, "runtime_binding_unavailable")

    return replace(capabilities, operations=operations)


async def require_capability(
    definition: AgentDefinition,
    operation: RuntimeOperationId | str,
    *,
    registry: RuntimeRegistry,
    run: Any | None = None,
) -> RuntimeOperationDescriptor:
    """Resolve one operation and fail before an adapter call when unavailable."""

    operation_id = operation.value if isinstance(operation, RuntimeOperationId) else str(operation)
    capabilities = await resolve_capabilities(definition, registry=registry, run=run)
    descriptor = capabilities.operations.get(operation_id)
    if descriptor is None or descriptor.support is RuntimeSupportLevel.UNSUPPORTED:
        raise RuntimeError.capability_unsupported(
            operation_id=operation_id,
            framework=definition.framework,
            builder_id=definition.builder_id,
            explanation="The runtime does not provide this operation",
        )
    if not descriptor.enabled:
        raise RuntimeError.capability_unavailable(
            operation_id=operation_id,
            framework=definition.framework,
            builder_id=definition.builder_id,
            support_level=descriptor.support.value,
            disabled_reason=str(descriptor.disabled_reason or "runtime_capability_unavailable"),
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
        "capabilities": capabilities.to_dict() if capabilities is not None else None,
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


async def discover_adapter_capabilities(
    adapter: Any,
    definition: AgentDefinition,
) -> tuple[RuntimeCapabilities | None, dict[str, Any] | None]:
    try:
        capabilities = await adapter.deployment_capabilities()
        operations = dict(capabilities.operations)
        for operation_id, descriptor in operations.items():
            if not descriptor.enabled or descriptor.owner is RuntimeOperationOwner.PRODUCT:
                continue
            method_name = OPERATION_METHODS.get(operation_id)
            if method_name is None:
                operations[operation_id] = _disabled(descriptor, "adapter_operation_unmapped")
                continue
            method = getattr(adapter, method_name, None)
            declared_method = getattr(type(adapter), method_name, None)
            base_method = getattr(AgentRuntimeAdapter, method_name, None)
            if method is None or declared_method is None or declared_method is base_method:
                operations[operation_id] = _disabled(descriptor, "adapter_operation_unimplemented")
        return replace(capabilities, operations=operations), None
    except RuntimeError as exc:
        return None, exc.to_dict()
    except Exception as exc:
        return None, RuntimeError.from_exception(
            exc,
            code="runtime_capability_discovery_failed",
            safe_message="Runtime capability discovery failed",
            details={"framework": adapter.framework, "builder_id": adapter.builder_id},
        ).to_dict()
