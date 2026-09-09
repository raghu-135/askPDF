from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.runtime.adapter import AgentRuntimeAdapter
from app.runtime.capability_resolver import (
    capabilities_for_definition,
    resolve_definition_capability_resolution,
    capability_envelope,
    discover_adapter_capabilities,
    require_capability,
    resolve_capabilities,
)
from runtime_protocol.contracts import (
    AgentDefinition,
    RuntimeCapabilityDisabledReason,
    RuntimeCapabilities,
    RuntimeOperationId,
    RuntimeOperationOwner,
    RuntimeOperationDescriptor,
    RuntimeSupportLevel,
    conditional,
    native,
    unsupported,
)
from runtime_protocol.errors import RuntimeError
from app.runtime.product_capabilities import project_public_capabilities
from app.runtime.registry import RuntimeRegistry


class CapabilityAdapter:
    framework = "fake"
    builder_id = "fake_builder"
    supports_task_pause = True
    implemented_operations = frozenset({
        RuntimeOperationId.RUN_START,
        RuntimeOperationId.RUN_CANCEL,
        RuntimeOperationId.RUN_RESUME,
        RuntimeOperationId.RUN_APPROVAL_RESPOND,
        RuntimeOperationId.RUN_INSPECT_STATE,
        RuntimeOperationId.RUN_REPLAY,
        RuntimeOperationId.RUN_SEND_FOLLOWUP,
    })

    def __init__(self, *, unsupported=()):
        self.calls = {"cancel": 0, "resume": 0, "replay": 0, "inspect_state": 0}
        self.unsupported = set(unsupported)
        self.capability_definition_ids = []
        self.deployment_capability_calls = 0

    async def capabilities(self, definition):
        self.capability_definition_ids.append(definition.definition_id)
        operations = {
                RuntimeOperationId.RUN_START: native(),
                RuntimeOperationId.RUN_CANCEL: native(),
                RuntimeOperationId.RUN_RESUME: native(),
                RuntimeOperationId.RUN_APPROVAL_RESPOND: native(),
                RuntimeOperationId.TASK_START: conditional(owner=RuntimeOperationOwner.PRODUCT, enabled=True),
                RuntimeOperationId.TASK_PAUSE: conditional(owner=RuntimeOperationOwner.PRODUCT, enabled=True),
                RuntimeOperationId.TASK_RESUME: conditional(owner=RuntimeOperationOwner.PRODUCT, enabled=True),
                RuntimeOperationId.TASK_CANCEL: conditional(owner=RuntimeOperationOwner.PRODUCT, enabled=True),
                RuntimeOperationId.TASK_RETRY: conditional(owner=RuntimeOperationOwner.PRODUCT, enabled=True),
                RuntimeOperationId.RUN_INSPECT_STATE: native(),
                RuntimeOperationId.RUN_REPLAY: unsupported(),
                RuntimeOperationId.RUN_STEER_LIVE: unsupported(),
            }
        for operation in self.unsupported:
            operations[RuntimeOperationId(operation)] = unsupported()
        return RuntimeCapabilities(operations=operations)

    async def deployment_capabilities(self):
        self.deployment_capability_calls += 1
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

    async def replay(self, request, checkpoint_id):
        self.calls["replay"] += 1
        return None

    async def inspect_state(self, request):
        self.calls["inspect_state"] += 1
        return {"state": {}}

    async def send_followup(self, request, input):
        return {"status": "queued"}


def test_public_capability_projection_excludes_spi_only_operations():
    capabilities = RuntimeCapabilities(operations={
        RuntimeOperationId.RUN_GET: native(),
        RuntimeOperationId.RUN_CANCEL: native(),
        RuntimeOperationId.RUN_REPLAY: native(),
        RuntimeOperationId.SUBAGENT_SEND: native(),
        RuntimeOperationId.TASK_START: conditional(owner=RuntimeOperationOwner.PRODUCT, enabled=True),
    })

    projected = project_public_capabilities(capabilities)

    assert set(projected.operations) == {
        RuntimeOperationId.RUN_GET,
        RuntimeOperationId.RUN_CANCEL,
        RuntimeOperationId.TASK_START,
    }
    assert RuntimeOperationId.RUN_REPLAY not in projected.operations
    assert RuntimeOperationId.SUBAGENT_SEND not in projected.operations


class HermesCapabilityAdapter(CapabilityAdapter):
    framework = "hermes"
    builder_id = "hermes_agent"


class UnavailableCapabilityAdapter(CapabilityAdapter):
    async def capabilities(self, definition):
        raise RuntimeError("runtime_unavailable", "Runtime is unavailable", retryable=True)

    async def deployment_capabilities(self):
        return await self.capabilities(AgentDefinition("deployment", self.framework, self.builder_id))


class BrokenCapabilityAdapter(CapabilityAdapter):
    async def capabilities(self, definition):
        raise AssertionError("adapter defect")


class ConfigurationInvalidCapabilityAdapter(CapabilityAdapter):
    async def capabilities(self, definition):
        raise RuntimeError("runtime_configuration_invalid", "invalid runtime configuration", retryable=False)


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
    assert adapter.capability_definition_ids[:2] == [task_definition.definition_id, non_task_definition.definition_id]
    assert adapter.deployment_capability_calls == 0


@pytest.mark.asyncio
async def test_definitions_on_one_deployment_keep_distinct_adapter_capabilities():
    class DefinitionAwareAdapter(CapabilityAdapter):
        async def capabilities(self, definition):
            capabilities = await super().capabilities(definition)
            operations = dict(capabilities.operations)
            operations[RuntimeOperationId.RUN_SEND_FOLLOWUP] = (
                native() if definition.definition_id == "followup-enabled" else unsupported()
            )
            return RuntimeCapabilities(operations=operations)

    adapter = DefinitionAwareAdapter()
    registry = RuntimeRegistry(adapters=[adapter])
    enabled = await capabilities_for_definition(
        AgentDefinition("followup-enabled", "fake", "fake_builder"), registry=registry
    )
    disabled = await capabilities_for_definition(
        AgentDefinition("followup-disabled", "fake", "fake_builder"), registry=registry
    )

    assert enabled.operations[RuntimeOperationId.RUN_SEND_FOLLOWUP].enabled is True
    assert disabled.operations[RuntimeOperationId.RUN_SEND_FOLLOWUP].support is RuntimeSupportLevel.UNSUPPORTED
    assert adapter.deployment_capability_calls == 0


@pytest.mark.asyncio
async def test_deployment_discovery_uses_deployment_declaration_and_adds_product_operations():
    adapter = CapabilityAdapter()
    capabilities, error = await discover_adapter_capabilities(adapter)

    assert error is None
    assert adapter.deployment_capability_calls == 1
    assert adapter.capability_definition_ids == ["deployment"]
    assert capabilities.operations[RuntimeOperationId.RUN_EVENTS].owner is RuntimeOperationOwner.PRODUCT
    assert capabilities.operations[RuntimeOperationId.TASK_START].owner is RuntimeOperationOwner.PRODUCT


@pytest.mark.asyncio
async def test_runtime_unavailable_deployment_disables_task_start():
    adapter = UnavailableCapabilityAdapter()
    capabilities, error = await discover_adapter_capabilities(adapter)

    assert capabilities is not None
    assert error["code"] == "runtime_unavailable"
    descriptor = capabilities.operations[RuntimeOperationId.TASK_START]
    assert descriptor.enabled is False
    assert descriptor.disabled_reason == RuntimeCapabilityDisabledReason.RUNTIME_UNAVAILABLE
    assert capabilities.operations[RuntimeOperationId.RUN_EVENTS].enabled is True
    assert capabilities.operations[RuntimeOperationId.RUN_EVENTS].owner is RuntimeOperationOwner.PRODUCT
    assert capabilities.operations[RuntimeOperationId.ARTIFACT_LIST].enabled is False
    assert capabilities.operations[RuntimeOperationId.ARTIFACT_LIST].owner is RuntimeOperationOwner.PRODUCT
    assert RuntimeOperationId.RUN_CANCEL not in capabilities.operations

    resolved = await resolve_capabilities(
        _definition(supports_long_running_tasks=True),
        registry=RuntimeRegistry(adapters=[adapter]),
    )
    assert resolved.operations[RuntimeOperationId.TASK_START].disabled_reason == RuntimeCapabilityDisabledReason.RUNTIME_UNAVAILABLE


@pytest.mark.asyncio
async def test_runtime_unavailable_disables_all_product_task_operations():
    capabilities, _ = await discover_adapter_capabilities(UnavailableCapabilityAdapter())

    for operation in (
        RuntimeOperationId.TASK_START,
        RuntimeOperationId.TASK_PAUSE,
        RuntimeOperationId.TASK_RESUME,
        RuntimeOperationId.TASK_CANCEL,
        RuntimeOperationId.TASK_RETRY,
    ):
        descriptor = capabilities.operations[operation]
        assert descriptor.enabled is False
        assert descriptor.disabled_reason == RuntimeCapabilityDisabledReason.RUNTIME_UNAVAILABLE


@pytest.mark.asyncio
async def test_capability_resolution_preserves_unavailable_runtime_error():
    resolution = await resolve_definition_capability_resolution(
        _definition(),
        registry=RuntimeRegistry(adapters=[UnavailableCapabilityAdapter()]),
    )

    assert resolution.runtime_available is False
    assert resolution.error is not None
    assert resolution.error["code"] == "runtime_unavailable"
    envelope = capability_envelope(
        capabilities=resolution.capabilities,
        resource="definition",
        runtime_id="fake:fake_builder",
        framework="fake",
        builder_id="fake_builder",
        error=resolution.error,
    )
    assert envelope["runtime_available"] is False
    assert envelope["error"]["code"] == "runtime_unavailable"
    assert "available" not in envelope


@pytest.mark.asyncio
async def test_configuration_failure_is_not_retryable_runtime_outage():
    adapter = ConfigurationInvalidCapabilityAdapter()
    resolution = await resolve_definition_capability_resolution(
        _definition(supports_long_running_tasks=True), registry=RuntimeRegistry(adapters=[adapter])
    )

    assert resolution.error["code"] == "runtime_configuration_invalid"
    assert resolution.error["safe_message"] == "invalid runtime configuration"
    assert resolution.error["retryable"] is False
    assert resolution.capabilities.operations[RuntimeOperationId.TASK_START].disabled_reason == RuntimeCapabilityDisabledReason.RUNTIME_CONFIGURATION_INVALID


@pytest.mark.asyncio
async def test_require_capability_preserves_discovery_failure_classification():
    with pytest.raises(RuntimeError) as caught:
        await require_capability(
            _definition(supports_long_running_tasks=True),
            RuntimeOperationId.TASK_START,
            registry=RuntimeRegistry(adapters=[ConfigurationInvalidCapabilityAdapter()]),
            run=SimpleNamespace(status="running"),
        )

    assert caught.value.code == "runtime_configuration_invalid"
    assert caught.value.retryable is False


@pytest.mark.asyncio
async def test_unexpected_capability_errors_are_not_classified_as_runtime_outages():
    definition = _definition()
    registry = RuntimeRegistry(adapters=[BrokenCapabilityAdapter()])

    with pytest.raises(AssertionError, match="adapter defect"):
        await capabilities_for_definition(definition, registry=registry)

    with pytest.raises(AssertionError, match="adapter defect"):
        await resolve_capabilities(definition, registry=registry)


@pytest.mark.asyncio
async def test_task_start_preserves_run_start_dependency_reason():
    adapter = CapabilityAdapter(unsupported=[RuntimeOperationId.RUN_START])
    capabilities = await resolve_capabilities(
        _definition(supports_long_running_tasks=True),
        registry=RuntimeRegistry(adapters=[adapter]),
    )

    descriptor = capabilities.operations[RuntimeOperationId.TASK_START]
    assert descriptor.enabled is False
    assert descriptor.disabled_reason == RuntimeCapabilityDisabledReason.RUNTIME_CAPABILITY_UNSUPPORTED


@pytest.mark.asyncio
async def test_task_start_is_admitted_before_a_runtime_run_exists():
    definition = _definition(supports_long_running_tasks=True)
    descriptor = await require_capability(
        definition,
        RuntimeOperationId.TASK_START,
        registry=RuntimeRegistry(adapters=[CapabilityAdapter()]),
    )

    assert descriptor.enabled is True


@pytest.mark.asyncio
async def test_artifact_listing_requires_a_task_owned_run():
    registry = RuntimeRegistry(adapters=[CapabilityAdapter()])
    definition = _definition(supports_long_running_tasks=True)

    deployment, _ = await discover_adapter_capabilities(registry.get(definition))
    assert deployment.operations[RuntimeOperationId.ARTIFACT_LIST].enabled is False

    definition_capabilities = await capabilities_for_definition(definition, registry=registry)
    assert definition_capabilities.operations[RuntimeOperationId.ARTIFACT_LIST].enabled is False

    run = SimpleNamespace(
        status="running",
        pending_interrupt_json=None,
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
    )
    taskless_run = await resolve_capabilities(definition, registry=registry, run=run)
    assert taskless_run.operations[RuntimeOperationId.ARTIFACT_LIST].enabled is False

    task_run = await resolve_capabilities(
        definition,
        registry=registry,
        run=run,
        task=SimpleNamespace(status="running"),
    )
    assert task_run.operations[RuntimeOperationId.ARTIFACT_LIST].enabled is True


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
async def test_clarification_run_is_terminal_for_runtime_controls():
    capabilities = await resolve_capabilities(
        _definition(),
        registry=RuntimeRegistry(adapters=[CapabilityAdapter()]),
        run=SimpleNamespace(
            status="clarification",
            pending_interrupt_json=None,
            runtime_binding_json={"binding_type": "fake"},
            runtime_binding_status="active",
        ),
    )

    assert capabilities.operations[RuntimeOperationId.RUN_CANCEL.value].disabled_reason == "run_terminal"


@pytest.mark.asyncio
async def test_task_and_run_states_are_resolved_independently():
    task_definition = _definition(supports_long_running_tasks=True)
    capabilities = await resolve_capabilities(
        task_definition,
        registry=RuntimeRegistry(adapters=[CapabilityAdapter()]),
        run=SimpleNamespace(
            status="completed",
            pending_interrupt_json=None,
            runtime_binding_json={},
            runtime_binding_status="active",
        ),
        task=SimpleNamespace(status="running"),
    )

    assert capabilities.operations[RuntimeOperationId.RUN_CANCEL.value].disabled_reason == "run_terminal"
    assert capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].enabled is True

    capabilities = await resolve_capabilities(
        task_definition,
        registry=RuntimeRegistry(adapters=[CapabilityAdapter()]),
        run=SimpleNamespace(
            status="running",
            pending_interrupt_json=None,
            runtime_binding_json={"binding_type": "fake"},
            runtime_binding_status="active",
        ),
        task=SimpleNamespace(status="failed"),
    )

    assert capabilities.operations[RuntimeOperationId.RUN_CANCEL.value].enabled is True
    assert capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].disabled_reason == "task_terminal"
    assert capabilities.operations[RuntimeOperationId.TASK_RETRY.value].enabled is True


@pytest.mark.asyncio
async def test_task_command_admission_uses_the_same_task_state_as_run_capabilities(monkeypatch):
    from app.api import agent_tasks as task_api

    task = SimpleNamespace(id="task-1", status="failed", workflow_id="definition-1")
    run = SimpleNamespace(
        id="run-1",
        workflow_id="definition-1",
        framework="fake",
        builder_id="fake_builder",
        definition_category="deep",
        status="running",
        pending_interrupt_json=None,
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
        resolved_spec_json={"runtime": {"features": {"supports_long_running_tasks": True}}},
    )
    workflow = SimpleNamespace(
        id="definition-1",
        framework="fake",
        builder_id="fake_builder",
        category="deep",
        name="Definition",
        metadata_json={},
        spec_json={"runtime": {"features": {"supports_long_running_tasks": False}}},
    )
    adapter = CapabilityAdapter()
    registry = RuntimeRegistry(adapters=[adapter])
    observed = {}

    class Repository:
        async def get_workflow(self, workflow_id, *, include_custom):
            assert workflow_id == workflow.id
            assert include_custom is True
            return workflow

    async def capture_require(definition, operation, *, registry, run=None, task=None, **kwargs):
        observed[operation] = await resolve_capabilities(
            definition, registry=registry, run=run, task=task, **kwargs,
        )

    monkeypatch.setattr(task_api, "AgentWorkflowRepository", Repository)
    monkeypatch.setattr(task_api.repository, "get_task_run", AsyncMock(return_value=run))
    monkeypatch.setattr(task_api, "get_runtime_registry", lambda: registry)
    monkeypatch.setattr(task_api, "require_capability", capture_require)

    await task_api._require_task_capability(task, "retry")

    definition = AgentDefinition(
        "definition-1", "fake", "fake_builder", category="deep",
        capabilities={"supports_long_running_tasks": True},
    )
    expected = await resolve_capabilities(
        definition, registry=registry, run=run, task=task,
    )
    assert observed[RuntimeOperationId.TASK_RETRY].operations == expected.operations


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
async def test_cancellation_pending_disables_run_and_task_cancel():
    run = SimpleNamespace(
        status="running",
        pending_interrupt_json=None,
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
        run_metadata_json={"runtime_started": True},
    )
    capabilities = await resolve_capabilities(
        _definition(supports_long_running_tasks=True),
        registry=RuntimeRegistry(adapters=[CapabilityAdapter()]),
        run=run,
        task=SimpleNamespace(status="cancelling"),
    )

    assert capabilities.operations[RuntimeOperationId.RUN_CANCEL].enabled is False
    assert capabilities.operations[RuntimeOperationId.TASK_CANCEL].enabled is False
    assert capabilities.operations[RuntimeOperationId.RUN_CANCEL].disabled_reason == RuntimeCapabilityDisabledReason.CANCELLATION_PENDING
    assert capabilities.operations[RuntimeOperationId.TASK_CANCEL].disabled_reason == RuntimeCapabilityDisabledReason.CANCELLATION_PENDING


@pytest.mark.asyncio
async def test_run_cancellation_marker_disables_repeated_cancellation():
    run = SimpleNamespace(
        status="running",
        pending_interrupt_json=None,
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
        run_metadata_json={"runtime_started": True, "cancel_requested": True},
    )
    capabilities = await resolve_capabilities(
        _definition(supports_long_running_tasks=True),
        registry=RuntimeRegistry(adapters=[CapabilityAdapter()]),
        run=run,
        task=SimpleNamespace(status="running"),
    )

    assert capabilities.operations[RuntimeOperationId.RUN_CANCEL].disabled_reason == RuntimeCapabilityDisabledReason.CANCELLATION_PENDING
    assert capabilities.operations[RuntimeOperationId.TASK_CANCEL].disabled_reason == RuntimeCapabilityDisabledReason.CANCELLATION_PENDING


@pytest.mark.asyncio
async def test_cancellation_is_available_after_pending_marker_clears_and_terminal_runs_remain_closed():
    definition = _definition(supports_long_running_tasks=True)
    registry = RuntimeRegistry(adapters=[CapabilityAdapter()])
    recovered = await resolve_capabilities(
        definition,
        registry=registry,
        run=SimpleNamespace(
            status="running",
            pending_interrupt_json=None,
            runtime_binding_json={"binding_type": "fake"},
            runtime_binding_status="active",
            run_metadata_json={"runtime_started": True, "cancel_requested": False},
        ),
        task=SimpleNamespace(status="running"),
    )
    assert recovered.operations[RuntimeOperationId.RUN_CANCEL].enabled is True
    assert recovered.operations[RuntimeOperationId.TASK_CANCEL].enabled is True

    confirmed = await resolve_capabilities(
        definition,
        registry=registry,
        run=SimpleNamespace(
            status="cancelled",
            pending_interrupt_json=None,
            runtime_binding_json={},
            runtime_binding_status="active",
            run_metadata_json={"runtime_started": True},
        ),
        task=SimpleNamespace(status="cancelling"),
    )
    assert confirmed.operations[RuntimeOperationId.RUN_CANCEL].disabled_reason == RuntimeCapabilityDisabledReason.RUN_TERMINAL


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", [
    RuntimeOperationId.RUN_STEER_LIVE,
    RuntimeOperationId.RUN_RESUME,
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
async def test_recovery_required_allows_retry_and_cancel_but_disables_live_runtime_controls():
    registry = RuntimeRegistry(adapters=[CapabilityAdapter()])
    definition = _definition(supports_long_running_tasks=True)
    run = SimpleNamespace(
        status="recovery_required", pending_interrupt_json=None,
        runtime_binding_json={"binding_type": "fake"}, runtime_binding_status="active",
        run_metadata_json={},
    )
    task = SimpleNamespace(status="recovery_required")

    capabilities = await resolve_capabilities(definition, registry=registry, run=run, task=task)

    assert capabilities.operations[RuntimeOperationId.TASK_RETRY.value].enabled is True
    assert capabilities.operations[RuntimeOperationId.TASK_CANCEL.value].enabled is True
    assert capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].enabled is False
    assert capabilities.operations[RuntimeOperationId.RUN_CANCEL.value].enabled is False
    assert capabilities.operations[RuntimeOperationId.RUN_CANCEL.value].disabled_reason == "recovery_required"


def test_recovery_required_task_is_deletable_without_being_normal_terminal():
    from app.services.agent_task_repository import DELETABLE_TASK_STATUSES, TERMINAL_TASK_STATUSES

    assert "recovery_required" in DELETABLE_TASK_STATUSES
    assert "recovery_required" not in TERMINAL_TASK_STATUSES


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
    supports_task_pause = True
    framework = "inherited"
    builder_id = "unsupported"

    async def capabilities(self, definition):
        return RuntimeCapabilities(operations={
            RuntimeOperationId.RUN_CANCEL: native(),
            RuntimeOperationId.TASK_PAUSE: conditional(owner=RuntimeOperationOwner.PRODUCT, enabled=True),
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
    )

    assert error is None
    assert capabilities.operations[RuntimeOperationId.RUN_CANCEL.value].enabled is False
    assert capabilities.operations[RuntimeOperationId.RUN_CANCEL.value].disabled_reason == "adapter_operation_unimplemented"
    assert capabilities.operations[RuntimeOperationId.TASK_PAUSE.value].enabled is True


@pytest.mark.asyncio
async def test_submitted_task_cancel_requires_effective_run_cancel():
    run = SimpleNamespace(
        status="running",
        pending_interrupt_json=None,
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
        run_metadata_json={"runtime_started": True},
    )
    capabilities = await resolve_capabilities(
        _definition(supports_long_running_tasks=True),
        registry=RuntimeRegistry(adapters=[CapabilityAdapter(unsupported=[RuntimeOperationId.RUN_CANCEL.value])]),
        run=run,
        task=SimpleNamespace(status="running"),
    )
    descriptor = capabilities.operations[RuntimeOperationId.TASK_CANCEL]
    assert descriptor.enabled is False
    assert descriptor.disabled_reason == RuntimeCapabilityDisabledReason.RUNTIME_CAPABILITY_UNSUPPORTED


@pytest.mark.asyncio
async def test_unsubmitted_task_cancel_remains_product_local():
    run = SimpleNamespace(
        status="running",
        pending_interrupt_json=None,
        runtime_binding_json={},
        runtime_binding_status="active",
        run_metadata_json={"runtime_started": False},
    )
    capabilities = await resolve_capabilities(
        _definition(supports_long_running_tasks=True),
        registry=RuntimeRegistry(adapters=[CapabilityAdapter(unsupported=[RuntimeOperationId.RUN_CANCEL.value])]),
        run=run,
        task=SimpleNamespace(status="queued"),
    )
    assert capabilities.operations[RuntimeOperationId.TASK_CANCEL].enabled is True
