"""Effective runtime capability resolution for API consumers."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from app.runtime.contracts import (
    AgentDefinition,
    RuntimeCapabilities,
    RuntimeOperationId,
    RuntimeOperationDescriptor,
    RuntimeSupportLevel,
)
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
    RuntimeOperationId.RUN_CONTINUE.value,
    RuntimeOperationId.RUN_CONTINUATION_CLEANUP.value,
})


def deployment_id(adapter: Any) -> str:
    return f"{adapter.framework}:{adapter.builder_id}"


def _disabled(
    descriptor: RuntimeOperationDescriptor,
    reason: str,
) -> RuntimeOperationDescriptor:
    if descriptor.support is RuntimeSupportLevel.UNSUPPORTED:
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
    status = str(getattr(run, "status", "") or "")
    pending = getattr(run, "pending_interrupt_json", None)
    has_pending = isinstance(pending, Mapping) and bool(pending)
    binding = getattr(run, "runtime_binding_json", None)
    binding_available = bool(binding) and str(getattr(run, "runtime_binding_status", "active")) == "active"

    if status in TERMINAL_RUN_STATES:
        for operation in ACTIVE_RUN_OPERATIONS:
            if operation in operations:
                operations[operation] = _disabled(operations[operation], "run_terminal")
    else:
        for operation in (RuntimeOperationId.RUN_RESUME.value, RuntimeOperationId.RUN_APPROVAL_RESPOND.value, RuntimeOperationId.INTERRUPT_RESPOND.value):
            if operation in operations and not has_pending:
                operations[operation] = _disabled(operations[operation], "no_pending_interrupt")

    if not binding_available and status not in TERMINAL_RUN_STATES:
        for operation in (
            RuntimeOperationId.RUN_CANCEL.value,
            RuntimeOperationId.RUN_RESUME.value,
            RuntimeOperationId.RUN_APPROVAL_RESPOND.value,
            RuntimeOperationId.INTERRUPT_RESPOND.value,
            RuntimeOperationId.RUN_INSPECT_STATE.value,
            RuntimeOperationId.RUN_CONTINUE.value,
            RuntimeOperationId.RUN_CONTINUATION_CLEANUP.value,
            RuntimeOperationId.TRACE_PROJECT.value,
        ):
            if operation in operations:
                operations[operation] = _disabled(operations[operation], "runtime_binding_unavailable")

    return replace(capabilities, operations=operations)


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
        return await adapter.capabilities(definition), None
    except RuntimeError as exc:
        return None, exc.to_dict()
    except Exception as exc:
        return None, RuntimeError.from_exception(
            exc,
            code="runtime_capability_discovery_failed",
            safe_message="Runtime capability discovery failed",
            details={"framework": adapter.framework, "builder_id": adapter.builder_id},
        ).to_dict()
