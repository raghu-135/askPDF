from types import SimpleNamespace

from app.runtime.catalog import definition_from_workflow
from app.runtime.contracts import (
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeCapabilities,
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeSupportLevel,
    RuntimeValidationIssue,
    RuntimeValidationResult,
    RuntimeArtifact,
    RuntimeTaskContext,
)
from app.runtime.langgraph_compat import (
    continuation_from_run,
    event_from_legacy,
    request_for_run,
    result_from_legacy,
)
from app.runtime.observability import normalize_runtime_event


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
        RuntimeOperationId.RUN_EVENTS.value: RuntimeOperationDescriptor(
            support=RuntimeSupportLevel.NATIVE,
            enabled=True,
        ),
        RuntimeOperationId.RUN_RESUME.value: RuntimeOperationDescriptor(
            support=RuntimeSupportLevel.CONDITIONAL,
            enabled=True,
            semantics="resume_from_interrupt",
        ),
    })

    assert request.to_dict()["continuation"]["payload"]["checkpoint_thread_id"] == "checkpoint-1"
    assert result.to_dict()["status"] == "completed"
    assert event.to_dict()["terminal"] is True
    assert capabilities.to_dict()["operations"]["run.resume"]["support"] == "conditional"
    assert list(capabilities.to_dict()["operations"]) == ["run.events", "run.resume"]


def test_runtime_operation_descriptor_rejects_invalid_enabled_states():
    import pytest

    with pytest.raises(ValueError):
        RuntimeOperationDescriptor(RuntimeSupportLevel.UNSUPPORTED, True)
    with pytest.raises(ValueError):
        RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, False)


def test_runtime_event_can_carry_an_opaque_continuation_binding():
    binding = ContinuationBinding(
        binding_type="hermes_session",
        runtime_version="hermes-gateway-1",
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


def test_langgraph_compat_round_trips_legacy_run_and_continuation():
    run = SimpleNamespace(
        id="run-1",
        thread_id="thread-1",
        workflow_id="router_rag_agent",
        framework="langgraph",
        builder_id="langgraph_graph",
        task_id=None,
        parent_run_id=None,
        checkpoint_thread_id="checkpoint-1",
        runtime_binding_version=1,
        runtime_binding_json={
            "binding_type": "langgraph_checkpoint",
            "binding_version": 1,
            "payload": {"checkpoint_thread_id": "checkpoint-1"},
        },
        run_metadata_json={},
    )

    binding = continuation_from_run(run)
    request = request_for_run(run, input={"question": "hello"})
    result = result_from_legacy({
        "status": "clarification",
        "clarification_options": ["one", "two"],
        "agent_run_id": "run-1",
    })
    event = event_from_legacy(
        {"event": "run.completed", "data": {"event_id": "runtime-event-1"}},
        run_id="run-1",
        sequence=2,
    )

    assert binding is not None
    assert binding.payload["checkpoint_thread_id"] == "checkpoint-1"
    assert request.continuation == binding
    assert request.input == {"question": "hello"}
    assert result.status == "clarification"
    assert result.clarification == {"options": ["one", "two"]}
    assert event.event_id == "runtime-event-1"
    assert event.terminal is True


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
