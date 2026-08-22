from types import SimpleNamespace

import pytest

from app.runtime.capability_resolver import capabilities_for_definition, require_capability, resolve_capabilities
from app.runtime.contracts import (
    AgentDefinition,
    RuntimeCapabilities,
    RuntimeOperationId,
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
                RuntimeOperationId.RUN_START.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, True),
                RuntimeOperationId.RUN_CANCEL.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, True),
                RuntimeOperationId.RUN_RESUME.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, True),
                RuntimeOperationId.RUN_PAUSE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.CONDITIONAL, True),
                RuntimeOperationId.RUN_RETRY.value: RuntimeOperationDescriptor(RuntimeSupportLevel.CONDITIONAL, True),
                RuntimeOperationId.RUN_INSPECT_STATE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, True),
                RuntimeOperationId.RUN_UPDATE_STATE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.UNSUPPORTED, False, disabled_reason="runtime_capability_unsupported"),
                RuntimeOperationId.RUN_REPLAY.value: RuntimeOperationDescriptor(RuntimeSupportLevel.UNSUPPORTED, False, disabled_reason="runtime_capability_unsupported"),
                RuntimeOperationId.RUN_STEER_LIVE.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.UNSUPPORTED,
                    False,
                    disabled_reason="runtime_capability_unsupported",
                ),
            }
        for operation in self.unsupported:
            operations[operation] = RuntimeOperationDescriptor(
                RuntimeSupportLevel.UNSUPPORTED,
                False,
                disabled_reason="runtime_capability_unsupported",
            )
        return RuntimeCapabilities(operations=operations)

    async def cancel(self, request):
        self.calls["cancel"] += 1
        return {"status": "cancel_requested"}

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
    assert capabilities.operations[RuntimeOperationId.RUN_PAUSE.value].enabled is True
    assert capabilities.operations[RuntimeOperationId.RUN_RETRY.value].enabled is False
    assert capabilities.operations[RuntimeOperationId.RUN_RETRY.value].disabled_reason == "run_not_retryable"

    failed = SimpleNamespace(
        status="failed",
        pending_interrupt_json=None,
        runtime_binding_json={},
        runtime_binding_status="active",
    )
    capabilities = await resolve_capabilities(task_definition, registry=registry, run=failed)
    assert capabilities.operations[RuntimeOperationId.RUN_PAUSE.value].enabled is False
    assert capabilities.operations[RuntimeOperationId.RUN_PAUSE.value].disabled_reason == "run_terminal"
    assert capabilities.operations[RuntimeOperationId.RUN_RETRY.value].enabled is True

    non_task_definition = _definition()
    capabilities = await resolve_capabilities(non_task_definition, registry=registry, run=running)
    assert capabilities.operations[RuntimeOperationId.RUN_PAUSE.value].disabled_reason == "definition_not_task_runtime"
    assert capabilities.operations[RuntimeOperationId.RUN_RETRY.value].disabled_reason == "definition_not_task_runtime"
