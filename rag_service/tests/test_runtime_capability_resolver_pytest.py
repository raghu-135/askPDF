from types import SimpleNamespace

import pytest

from app.runtime.adapter import AgentRuntimeAdapter
from app.runtime.capability_resolver import (
    OPERATION_METHODS,
    capabilities_for_definition,
    discover_adapter_capabilities,
    require_capability,
    resolve_capabilities,
)
from app.runtime.contracts import (
    AgentDefinition,
    RuntimeCapabilities,
    RuntimeOperationId,
    RuntimeOperationOwner,
    RuntimeOperationDescriptor,
    RuntimeSupportLevel,
)
from app.runtime.errors import RuntimeError
from app.runtime.registry import RuntimeRegistry


class CapabilityAdapter:
    framework = "fake"
    builder_id = "fake_builder"

    def __init__(self, *, unsupported=()):
        self.calls = {"cancel": 0, "resume": 0, "update_state": 0, "replay": 0, "inspect_state": 0}
        self.unsupported = set(unsupported)

    async def capabilities(self, definition):
        operations = {
                RuntimeOperationId.RUN_START.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, True),
                RuntimeOperationId.RUN_CANCEL.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, True),
                RuntimeOperationId.RUN_RESUME.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, True),
                RuntimeOperationId.RUN_APPROVAL_RESPOND.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, True),
                RuntimeOperationId.TASK_START.value: RuntimeOperationDescriptor(RuntimeSupportLevel.CONDITIONAL, RuntimeOperationOwner.PRODUCT, True),
                RuntimeOperationId.TASK_PAUSE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.CONDITIONAL, RuntimeOperationOwner.PRODUCT, True),
                RuntimeOperationId.TASK_RESUME.value: RuntimeOperationDescriptor(RuntimeSupportLevel.CONDITIONAL, RuntimeOperationOwner.PRODUCT, True),
                RuntimeOperationId.TASK_CANCEL.value: RuntimeOperationDescriptor(RuntimeSupportLevel.CONDITIONAL, RuntimeOperationOwner.PRODUCT, True),
                RuntimeOperationId.TASK_RETRY.value: RuntimeOperationDescriptor(RuntimeSupportLevel.CONDITIONAL, RuntimeOperationOwner.PRODUCT, True),
                RuntimeOperationId.RUN_INSPECT_STATE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, True),
                RuntimeOperationId.RUN_UPDATE_STATE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.UNSUPPORTED, RuntimeOperationOwner.RUNTIME, False, disabled_reason="runtime_capability_unsupported"),
                RuntimeOperationId.RUN_REPLAY.value: RuntimeOperationDescriptor(RuntimeSupportLevel.UNSUPPORTED, RuntimeOperationOwner.RUNTIME, False, disabled_reason="runtime_capability_unsupported"),
                RuntimeOperationId.RUN_STEER_LIVE.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.UNSUPPORTED,
                    RuntimeOperationOwner.RUNTIME,
                    False,
                    disabled_reason="runtime_capability_unsupported",
                ),
            }
        for operation in self.unsupported:
            operations[operation] = RuntimeOperationDescriptor(
                RuntimeSupportLevel.UNSUPPORTED,
                RuntimeOperationOwner.RUNTIME,
                False,
                disabled_reason="runtime_capability_unsupported",
            )
        return RuntimeCapabilities(operations=operations)

    async def deployment_capabilities(self):
        return await self.capabilities(AgentDefinition("deployment", self.framework, self.builder_id))

    async def cancel(self, request):
        self.calls["cancel"] += 1
        return {"status": "cancel_requested"}

    async def start(self, request, *, context=None, event_sink=None):
        return None

    async def respond_to_approval(self, request, response):
        return None

    async def resume(self, request, *, interrupt, context, event_sink=None):
        self.calls["resume"] += 1
        return None

    async def update_state(self, request, update):
        self.calls["update_state"] += 1
        return {"status": "updated"}

    async def replay(self, request, checkpoint_id):
        self.calls["replay"] += 1
        return None

    async def inspect_state(self, request):
        self.calls["inspect_state"] += 1
        return {"state": {}}


class HermesCapabilityAdapter(CapabilityAdapter):
    framework = "hermes"
    builder_id = "hermes_agent"


def _definition(**capabilities):
    return AgentDefinition(
        definition_id="definition-1",
        framework="fake",
        builder_id="fake_builder",
        capabilities=capabilities,
    )


@pytest.mark.asyncio
async def test_definition_policy_and_run_state_gate_operations():
    run = SimpleNamespace(
        status="running",
        pending_interrupt_json=None,
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
    )
    capabilities = await resolve_capabilities(
        _definition(disabled_operations=[RuntimeOperationId.RUN_START.value]),
        registry=RuntimeRegistry(adapters=[CapabilityAdapter()]),
        run=run,
    )

    assert capabilities.operations[RuntimeOperationId.RUN_START.value].enabled is False
    assert capabilities.operations[RuntimeOperationId.RUN_START.value].disabled_reason == "definition_policy"
    assert capabilities.operations[RuntimeOperationId.RUN_RESUME.value].enabled is False
    assert capabilities.operations[RuntimeOperationId.RUN_RESUME.value].disabled_reason == "no_pending_interrupt"
    assert capabilities.operations[RuntimeOperationId.RUN_STEER_LIVE.value].support is RuntimeSupportLevel.UNSUPPORTED
    assert capabilities.operations[RuntimeOperationId.RUN_STEER_LIVE.value].enabled is False
    assert capabilities.operations[RuntimeOperationId.RUN_EVENTS.value].owner is RuntimeOperationOwner.PRODUCT


def test_operation_method_mapping_uses_runtime_operation_values():
    assert OPERATION_METHODS[RuntimeOperationId.RUN_START.value] == "start"
    assert OPERATION_METHODS[RuntimeOperationId.RUN_APPROVAL_RESPOND.value] == "respond_to_approval"


@pytest.mark.asyncio
async def test_run_capabilities_disable_start_for_an_existing_run():
    capabilities = await resolve_capabilities(
        _definition(),
        registry=RuntimeRegistry(adapters=[CapabilityAdapter()]),
        run=SimpleNamespace(
            status="running",
            pending_interrupt_json=None,
            runtime_binding_json={"binding_type": "fake"},
            runtime_binding_status="active",
        ),
    )

    descriptor = capabilities.operations[RuntimeOperationId.RUN_START.value]
    assert descriptor.enabled is False
    assert descriptor.disabled_reason == "run_already_created"


@pytest.mark.asyncio
async def test_run_capabilities_keep_start_enabled_for_a_fresh_task_run():
    capabilities = await resolve_capabilities(
        _definition(),
        registry=RuntimeRegistry(adapters=[CapabilityAdapter()]),
        run=SimpleNamespace(
            status="running",
            pending_interrupt_json=None,
            runtime_binding_json={},
            runtime_binding_status="active",
            _fresh_runtime_run=True,
        ),
    )

    descriptor = capabilities.operations[RuntimeOperationId.RUN_START.value]
    assert descriptor.enabled is True


@pytest.mark.asyncio
async def test_definition_capabilities_are_requested_from_adapter_and_drive_task_operations():
    adapter = CapabilityAdapter()
    registry = RuntimeRegistry(adapters=[adapter])

    task_definition = _definition(supports_long_running_tasks=True)
    task_capabilities = await capabilities_for_definition(task_definition, registry=registry)
    assert task_capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].enabled is True
    assert task_capabilities.operations[RuntimeOperationId.TASK_RETRY.value].enabled is True

    non_task_definition = _definition(supports_long_running_tasks=False)
    non_task_capabilities = await capabilities_for_definition(non_task_definition, registry=registry)
    assert non_task_capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].enabled is False
    assert non_task_capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].disabled_reason == "definition_not_task_runtime"
    resolved = await resolve_capabilities(non_task_definition, registry=registry, run=SimpleNamespace(
        status="running",
        pending_interrupt_json=None,
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
    ))
    assert resolved.operations[RuntimeOperationId.TASK_PAUSE.value].disabled_reason == "definition_not_task_runtime"
    assert "run.continue" not in task_capabilities.operations


@pytest.mark.asyncio
async def test_hermes_live_steering_remains_disabled_at_all_capability_levels():
    registry = RuntimeRegistry(adapters=[HermesCapabilityAdapter()])
    definition = AgentDefinition(
        definition_id="hermes_rag_agent",
        framework="hermes",
        builder_id="hermes_agent",
    )
    run = SimpleNamespace(
        status="running",
        pending_interrupt_json=None,
        runtime_binding_json={"binding_type": "hermes_session"},
        runtime_binding_status="active",
    )

    runtime_capabilities = await resolve_capabilities(definition, registry=registry)
    definition_capabilities = await capabilities_for_definition(definition, registry=registry)
    run_capabilities = await resolve_capabilities(definition, registry=registry, run=run)

    for capabilities in (runtime_capabilities, definition_capabilities, run_capabilities):
        descriptor = capabilities.operations[RuntimeOperationId.RUN_STEER_LIVE.value]
        assert descriptor.support is RuntimeSupportLevel.UNSUPPORTED
        assert descriptor.enabled is False
        assert descriptor.disabled_reason == "runtime_capability_unsupported"


@pytest.mark.asyncio
async def test_terminal_run_disables_active_controls_without_mutating_source():
    run = SimpleNamespace(
        status="completed",
        pending_interrupt_json={"interrupt_id": "i-1"},
        runtime_binding_json={},
        runtime_binding_status="active",
    )
    capabilities = await resolve_capabilities(
        _definition(),
        registry=RuntimeRegistry(adapters=[CapabilityAdapter()]),
        run=run,
    )

    cancel = capabilities.operations[RuntimeOperationId.RUN_CANCEL.value]
    assert cancel.enabled is False
    assert cancel.disabled_reason == "run_terminal"
    assert capabilities.operations[RuntimeOperationId.RUN_APPROVAL_RESPOND.value].disabled_reason == "run_terminal"
    assert run.status == "completed"
    assert run.pending_interrupt_json == {"interrupt_id": "i-1"}


@pytest.mark.asyncio
async def test_require_capability_rejects_unsupported_operation_before_adapter_call():
    with pytest.raises(RuntimeError) as caught:
        await require_capability(
            _definition(),
            RuntimeOperationId.RUN_STEER_LIVE,
            registry=RuntimeRegistry(adapters=[CapabilityAdapter()]),
        )

    assert caught.value.code == "runtime_capability_unsupported"
    assert caught.value.details["operation_id"] == "run.steer_live"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", [
    RuntimeOperationId.RUN_STEER_LIVE,
    RuntimeOperationId.RUN_RESUME,
    RuntimeOperationId.RUN_UPDATE_STATE,
    RuntimeOperationId.RUN_REPLAY,
])
async def test_unsupported_operations_are_rejected_without_runtime_invocation(operation):
    adapter = CapabilityAdapter(unsupported={operation.value})
    with pytest.raises(RuntimeError) as caught:
        await require_capability(
            _definition(),
            operation,
            registry=RuntimeRegistry(adapters=[adapter]),
        )

    assert caught.value.code == "runtime_capability_unsupported"
    assert caught.value.details["support_level"] == "unsupported"
    assert all(value == 0 for value in adapter.calls.values())


@pytest.mark.asyncio
async def test_supported_but_ineligible_operation_is_rejected_without_runtime_invocation():
    adapter = CapabilityAdapter()
    terminal = SimpleNamespace(
        status="completed",
        pending_interrupt_json=None,
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
    )
    with pytest.raises(RuntimeError) as caught:
        await require_capability(
            _definition(),
            RuntimeOperationId.RUN_CANCEL,
            registry=RuntimeRegistry(adapters=[adapter]),
            run=terminal,
        )

    assert caught.value.code == "runtime_capability_unavailable"
    assert caught.value.details["disabled_reason"] == "run_terminal"
    assert all(value == 0 for value in adapter.calls.values())


@pytest.mark.asyncio
async def test_task_lifecycle_operations_require_task_definition_and_eligible_state():
    registry = RuntimeRegistry(adapters=[CapabilityAdapter()])
    task_definition = _definition(supports_long_running_tasks=True)
    running = SimpleNamespace(
        status="running",
        pending_interrupt_json=None,
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
    )
    capabilities = await resolve_capabilities(task_definition, registry=registry, run=running)
    assert capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].enabled is True
    assert capabilities.operations[RuntimeOperationId.TASK_RETRY.value].enabled is False
    assert capabilities.operations[RuntimeOperationId.TASK_RETRY.value].disabled_reason == "task_not_retryable"

    failed = SimpleNamespace(
        status="failed",
        pending_interrupt_json=None,
        runtime_binding_json={},
        runtime_binding_status="active",
    )
    capabilities = await resolve_capabilities(task_definition, registry=registry, run=failed)
    assert capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].enabled is False
    assert capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].disabled_reason == "task_terminal"
    assert capabilities.operations[RuntimeOperationId.TASK_RETRY.value].enabled is False
    assert capabilities.operations[RuntimeOperationId.TASK_RETRY.value].disabled_reason == "task_terminal"

    non_task_definition = _definition()
    capabilities = await resolve_capabilities(non_task_definition, registry=registry, run=running)
    assert capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].disabled_reason == "definition_not_task_runtime"
    assert capabilities.operations[RuntimeOperationId.TASK_RETRY.value].disabled_reason == "definition_not_task_runtime"


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt_status", ["resolved", "rejected", "expired"])
async def test_only_explicit_pending_interrupts_enable_the_declared_response_operation(interrupt_status):
    registry = RuntimeRegistry(adapters=[CapabilityAdapter()])
    definition = _definition()
    run = SimpleNamespace(
        status="awaiting_human",
        pending_interrupt_json={
            "status": interrupt_status,
            "response_operation": RuntimeOperationId.RUN_APPROVAL_RESPOND.value,
        },
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
    )

    capabilities = await resolve_capabilities(definition, registry=registry, run=run)

    for operation in (
        RuntimeOperationId.RUN_RESUME,
        RuntimeOperationId.RUN_APPROVAL_RESPOND,
    ):
        assert capabilities.operations[operation.value].enabled is False
        assert capabilities.operations[operation.value].disabled_reason == "no_pending_interrupt"


@pytest.mark.asyncio
async def test_pending_interrupt_enables_only_its_valid_response_operation():
    registry = RuntimeRegistry(adapters=[CapabilityAdapter()])
    run = SimpleNamespace(
        status="awaiting_human",
        pending_interrupt_json={
            "status": "pending",
            "response_operation": RuntimeOperationId.RUN_APPROVAL_RESPOND.value,
        },
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
    )

    capabilities = await resolve_capabilities(_definition(), registry=registry, run=run)

    assert capabilities.operations[RuntimeOperationId.RUN_APPROVAL_RESPOND.value].enabled is True
    assert capabilities.operations[RuntimeOperationId.RUN_RESUME.value].disabled_reason == "no_pending_interrupt"


@pytest.mark.asyncio
async def test_completed_run_with_stale_pending_payload_disables_all_active_controls():
    registry = RuntimeRegistry(adapters=[CapabilityAdapter()])
    run = SimpleNamespace(
        status="completed",
        pending_interrupt_json={
            "status": "pending",
            "response_operation": RuntimeOperationId.RUN_RESUME.value,
        },
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
    )

    capabilities = await resolve_capabilities(_definition(), registry=registry, run=run)

    for operation in (
        RuntimeOperationId.RUN_CANCEL,
        RuntimeOperationId.RUN_RESUME,
        RuntimeOperationId.RUN_APPROVAL_RESPOND,
    ):
        assert capabilities.operations[operation.value].disabled_reason == "run_terminal"


class InheritedUnsupportedAdapter(AgentRuntimeAdapter):
    framework = "inherited"
    builder_id = "unsupported"

    async def capabilities(self, definition):
        return RuntimeCapabilities(operations={
            RuntimeOperationId.RUN_CANCEL.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, True),
                RuntimeOperationId.TASK_PAUSE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.CONDITIONAL, RuntimeOperationOwner.PRODUCT, True),
        })

    async def validate(self, definition, spec, *, options=None):
        raise NotImplementedError

    async def start(self, request, *, context, event_sink=None):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_discovery_rejects_enabled_operation_that_only_inherits_base_unsupported_method():
    adapter = InheritedUnsupportedAdapter()
    capabilities, error = await discover_adapter_capabilities(
        adapter,
        AgentDefinition("definition-1", adapter.framework, adapter.builder_id),
    )

    assert error is None
    assert capabilities.operations[RuntimeOperationId.RUN_CANCEL.value].enabled is False
    assert capabilities.operations[RuntimeOperationId.RUN_CANCEL.value].disabled_reason == "adapter_operation_unimplemented"
    assert capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].enabled is True
