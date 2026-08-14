from types import SimpleNamespace

from app.runtime.catalog import definition_from_workflow
from app.runtime.contracts import (
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeCapabilities,
)
from app.runtime.langgraph_compat import (
    continuation_from_run,
    event_from_legacy,
    request_for_run,
    result_from_legacy,
)


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
    capabilities = RuntimeCapabilities(streaming=True, resume=True, native_checkpoints=True)

    assert request.to_dict()["continuation"]["payload"]["checkpoint_thread_id"] == "checkpoint-1"
    assert result.to_dict()["status"] == "completed"
    assert event.to_dict()["terminal"] is True
    assert capabilities.to_dict()["resume"] is True


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
