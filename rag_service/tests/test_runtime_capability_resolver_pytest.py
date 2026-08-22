from types import SimpleNamespace

import pytest

from app.runtime.capability_resolver import require_capability, resolve_capabilities
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

    async def capabilities(self, definition):
        return RuntimeCapabilities(
            operations={
                RuntimeOperationId.RUN_START.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, True),
                RuntimeOperationId.RUN_CANCEL.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, True),
                RuntimeOperationId.RUN_RESUME.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, True),
                RuntimeOperationId.RUN_PAUSE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.CONDITIONAL, True),
                RuntimeOperationId.RUN_RETRY.value: RuntimeOperationDescriptor(RuntimeSupportLevel.CONDITIONAL, True),
                RuntimeOperationId.RUN_INSPECT_STATE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, True),
                RuntimeOperationId.RUN_STEER_LIVE.value: RuntimeOperationDescriptor(
                    RuntimeSupportLevel.UNSUPPORTED,
                    False,
                    disabled_reason="runtime_capability_unsupported",
                ),
            }
        )


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
