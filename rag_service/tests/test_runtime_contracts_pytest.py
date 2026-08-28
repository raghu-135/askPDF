from types import SimpleNamespace

import pytest

from app.runtime.adapter import AgentRuntimeAdapter, RuntimeExecutionContext
from app.runtime.errors import RuntimeError
from app.runtime.catalog import (
    continuation_from_run,
    definition_from_run,
    definition_from_workflow,
    event_from_source,
    request_from_run,
    result_to_product_payload,
)
from app.runtime.contracts import (
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeApprovalResponse,
    RuntimeSteeringInput,
    RuntimeCapabilities,
    RuntimeCapabilitySemantics,
    RuntimeFeatureDescriptor,
    RuntimeFeatureId,
    RuntimeSupportLevel,
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeOperationOwner,
    RuntimeSupportLevel,
    RuntimeValidationIssue,
    RuntimeValidationResult,
    RuntimeArtifact,
    RuntimeTaskContext,
    validated_disabled_operation_ids,
)
from app.runtime.observability import normalize_runtime_event
from app.agent_workflows.interrupts import AgentRunInterruptError, normalize_pending_interrupt_payload
from app.runtime.product_capabilities import project_public_capabilities


def test_neutral_contracts_are_frozen_and_json_compatible():
    binding = ContinuationBinding(
        binding_type="langgraph_checkpoint",
        payload={"checkpoint_thread_id": "checkpoint-1"},
    )
    request = AgentRuntimeRequest(
        run_id="run-1",
        thread_id="thread-1",
        definition_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        continuation=binding,
    )
    result = AgentRuntimeResult(status="completed", output={"answer": "ok"}, continuation=binding)
    event = AgentRuntimeEvent(
        event_id="event-1",
        run_id="run-1",
        sequence=1,
        kind="run.completed",
        terminal=True,
    )
    capabilities = RuntimeCapabilities(operations={
        RuntimeOperationId.RUN_EVENTS: RuntimeOperationDescriptor(
            support=RuntimeSupportLevel.NATIVE,
            owner=RuntimeOperationOwner.PRODUCT,
            enabled=True,
        ),
        RuntimeOperationId.RUN_RESUME: RuntimeOperationDescriptor(
            support=RuntimeSupportLevel.CONDITIONAL,
            owner=RuntimeOperationOwner.RUNTIME,
            enabled=True,
        semantics=RuntimeCapabilitySemantics.RESUME_FROM_INTERRUPT,
        ),
    })

    assert request.to_dict()["continuation"]["payload"]["checkpoint_thread_id"] == "checkpoint-1"
    assert result.to_dict()["status"] == "completed"
    assert event.to_dict()["terminal"] is True
    assert capabilities.to_dict()["operations"]["run.resume"]["support"] == "conditional"
    assert capabilities.to_dict()["operations"]["run.resume"]["owner"] == "runtime"
    assert list(capabilities.to_dict()["operations"]) == ["run.events", "run.resume"]


def test_runtime_operation_descriptor_rejects_invalid_enabled_states():
    with pytest.raises(ValueError):
        RuntimeOperationDescriptor(RuntimeSupportLevel.UNSUPPORTED, RuntimeOperationOwner.RUNTIME, True)
    with pytest.raises(ValueError):
        RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, False)


def test_runtime_operation_descriptor_requires_a_typed_owner():
    with pytest.raises(TypeError):
        RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, enabled=True)
    with pytest.raises(ValueError):
        RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, "runtime", True)


@pytest.mark.parametrize("response_operation", [None, "", "interrupt.respond"])
def test_pending_interrupt_requires_an_implemented_response_operation(response_operation):
    payload = {"interrupt_id": "interrupt-1"}
    if response_operation is not None:
        payload["response_operation"] = response_operation

    with pytest.raises(AgentRunInterruptError) as caught:
        normalize_pending_interrupt_payload(payload)

    assert caught.value.code == "interrupt_response_operation_invalid"


def test_public_capability_projection_preserves_approval_response():
    capabilities = RuntimeCapabilities(operations={
        RuntimeOperationId.RUN_APPROVAL_RESPOND: RuntimeOperationDescriptor(
            support=RuntimeSupportLevel.NATIVE,
            owner=RuntimeOperationOwner.RUNTIME,
            enabled=True,
        ),
    })

    projected = project_public_capabilities(capabilities)

    assert projected.operations[RuntimeOperationId.RUN_APPROVAL_RESPOND].enabled is True


def test_continuation_requires_authoritative_runtime_binding():
    run = SimpleNamespace(id="run-1", checkpoint_thread_id="checkpoint-1", runtime_binding_json={})

    assert continuation_from_run(run) is None


@pytest.mark.asyncio
async def test_optional_adapter_methods_have_structured_unsupported_defaults():
    class MinimalAdapter(AgentRuntimeAdapter):
        framework = "minimal"
        builder_id = "minimal_builder"

        async def capabilities(self, definition):
            return RuntimeCapabilities()

        async def validate(self, definition, spec, *, options=None):
            return RuntimeValidationResult(valid=True)

        async def start(self, request, *, context, event_sink=None):
            return AgentRuntimeResult(status="completed")

    adapter = MinimalAdapter()
    request = AgentRuntimeRequest("run-1", "thread-1", "definition-1", "minimal", "minimal_builder")
    operations = (
        ("run.get", lambda: adapter.get_run(request)),
        ("run.list", lambda: adapter.list_runs(thread_id="thread-1")),
        ("run.wait", lambda: adapter.wait(request)),
        ("run.events", lambda: adapter.stream_events(request)),
        ("run.resume", lambda: adapter.resume(request, interrupt={}, context=RuntimeExecutionContext())),
        ("runtime_continuation_unavailable", lambda: adapter.continue_run(request, context=RuntimeExecutionContext())),
        ("run.cancel", lambda: adapter.cancel(request)),
        ("run.approval.respond", lambda: adapter.respond_to_approval(request, RuntimeApprovalResponse("approve", scope="once"))),
        ("run.send_followup", lambda: adapter.send_followup(request, {})),
        ("run.interrupt_with_input", lambda: adapter.interrupt_with_input(request, {})),
        ("run.steer_live", lambda: adapter.steer_live(request, RuntimeSteeringInput("focus"))),
        ("run.inspect_state", lambda: adapter.inspect_state(request)),
        ("run.update_state", lambda: adapter.update_state(request, {})),
        ("run.replay", lambda: adapter.replay(request, "checkpoint-1")),
        ("run.fork", lambda: adapter.fork(request, "checkpoint-1")),
        ("subagent.list", lambda: adapter.list_subagents(request)),
        ("subagent.send", lambda: adapter.send_to_subagent(request, "subagent-1", {})),
        ("subagent.cancel", lambda: adapter.cancel_subagent(request, "subagent-1")),
        ("artifact.list", lambda: adapter.list_artifacts(request)),
        ("run.continuation.cleanup", lambda: adapter.delete_continuation(ContinuationBinding("test", {}))),
        ("trace.project", lambda: adapter.project_trace([], run_id="run-1")),
    )
    for operation_id, invoke in operations:
        with pytest.raises(RuntimeError) as caught:
            await invoke()
        if operation_id == "runtime_continuation_unavailable":
            assert caught.value.code == operation_id
            continue
        assert caught.value.code == "runtime_capability_unsupported"
        assert caught.value.retryable is False
        assert caught.value.details == {
            "operation_id": operation_id,
            "framework": "minimal",
            "builder_id": "minimal_builder",
            "support_level": "unsupported",
            "explanation": caught.value.details["explanation"],
        }


def test_runtime_event_can_carry_an_opaque_continuation_binding():
    binding = ContinuationBinding(
        binding_type="hermes_session",
        payload={"session_id": "session-1", "upstream_run_id": "hermes-run-7"},
    )
    event = AgentRuntimeEvent(
        event_id="event-1",
        run_id="run-1",
        sequence=1,
        kind="runtime.session_started",
        continuation=binding,
    )
    assert event.to_dict()["continuation"]["payload"]["upstream_run_id"] == "hermes-run-7"


def test_catalog_identity_is_concrete_and_category_is_metadata_only():
    workflow = SimpleNamespace(
        id="router_rag_agent",
        name="Router Agent",
        version=1,
        framework="langgraph",
        builder_id="langgraph_graph",
        category="router",
        metadata_json={"builtin_key": "router_rag_agent"},
        spec_json={"runtime": {"features": {"supports_replans": False}}},
    )

    definition = definition_from_workflow(workflow)

    assert definition.definition_id == "router_rag_agent"
    assert definition.framework == "langgraph"
    assert definition.builder_id == "langgraph_graph"
    assert definition.category == "router"
    assert definition.capabilities == {"supports_replans": False}


def test_definition_rejects_unknown_disabled_operations():
    with pytest.raises(ValueError, match="unknown operations: run.not_real"):
        validated_disabled_operation_ids(["run.not_real"])


def test_run_identity_and_typed_projection_round_trip():
    run = SimpleNamespace(
        id="run-1",
        thread_id="thread-1",
        workflow_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        task_id=None,
        parent_run_id=None,
        checkpoint_thread_id="checkpoint-1",
        runtime_binding_json={
            "binding_type": "langgraph_checkpoint",
            "payload": {"checkpoint_thread_id": "checkpoint-1"},
        },
        run_metadata_json={},
    )

    binding = continuation_from_run(run)
    request = request_from_run(run, input={"question": "hello"})
    result = AgentRuntimeResult(status="clarification", clarification={"options": ["one", "two"]})
    event = event_from_source(
        {"event": "run.completed", "data": {"event_id": "runtime-event-1"}},
        run_id="run-1",
        sequence=2,
    )

    assert binding is not None
    assert binding.payload["checkpoint_thread_id"] == "checkpoint-1"
    assert request.continuation == binding
    assert request.input == {"question": "hello"}
    assert result.status == "clarification"
    assert result_to_product_payload(result)["clarification_options"] == ["one", "two"]
    assert event.event_id == "runtime-event-1"
    assert event.terminal is True


def test_workflow_and_run_definition_metadata_are_identical():
    spec = {
        "runtime": {"features": {"supports_replans": False}},
        "config": {
            "allowed_tool_ids": ["search_documents"],
            "task_policy": {"profiles": ["research"]},
        },
    }
    workflow = SimpleNamespace(
        id="router_rag_agent",
        name="Router Agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        category="router",
        metadata_json={},
        spec_json=spec,
    )
    run = SimpleNamespace(
        id="run-1",
        workflow_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        definition_category="router",
        resolved_spec_json=spec,
    )

    workflow_definition = definition_from_workflow(workflow)
    run_definition = definition_from_run(run)
    assert run_definition.definition_metadata == workflow_definition.definition_metadata
    assert run_definition.capabilities == workflow_definition.capabilities


def test_validation_contract_is_json_compatible():
    result = RuntimeValidationResult(
        valid=False,
        issues=(RuntimeValidationIssue(code="invalid_workflow", message="bad spec", path="config.graph"),),
        runtime_metadata={"framework": "langgraph"},
    )

    assert result.to_dict()["issues"][0]["path"] == "config.graph"
    assert result.to_dict()["valid"] is False


def test_runtime_task_context_and_artifact_are_json_compatible():
    artifact = RuntimeArtifact(kind="intermediate_report", content="report", todo_id="todo-1")
    context = RuntimeTaskContext(
        task_id="task-1",
        objective="research",
        todos=({"id": "todo-1", "status": "pending"},),
        artifact_manifests=(artifact.to_dict(),),
        artifact_contents={artifact.artifact_id or "runtime": "report"},
    )

    assert artifact.to_dict()["kind"] == "intermediate_report"
    assert context.to_dict()["todos"][0]["id"] == "todo-1"


def test_langgraph_node_events_normalize_to_topology_linked_operations():
    kind, payload = normalize_runtime_event(
        "node.completed",
        {"node_id": "planner", "node_type": "planner", "visit_index": 2, "elapsed_ms": 17},
    )

    assert kind == "operation.completed"
    assert payload["operation_id"] == "planner"
    assert payload["operation_type"] == "planner"
    assert payload["visit_index"] == 2
    assert payload["topology_ref"] == {"kind": "graph_node", "id": "planner"}


def test_runtime_operations_remain_topology_optional():
    kind, payload = normalize_runtime_event(
        "operation.started",
        {"operation_id": "hermes_session", "operation_type": "agent_session"},
    )

    assert kind == "operation.started"
    assert payload["operation_id"] == "hermes_session"
    assert "topology_ref" not in payload


def test_feature_identifiers_and_operation_metadata_are_closed_vocabularies():
    descriptor = RuntimeFeatureDescriptor(RuntimeSupportLevel.NATIVE, True)
    assert RuntimeCapabilities(features={RuntimeFeatureId.TOOLS: descriptor}).to_dict()["features"] == {
        "tools": {"support": "native", "enabled": True, "disabled_reason": None}
    }
    with pytest.raises(ValueError, match="RuntimeFeatureId"):
        RuntimeCapabilities(features={"tools": descriptor})
