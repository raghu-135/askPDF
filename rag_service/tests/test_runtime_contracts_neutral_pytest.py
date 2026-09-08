"""Control-plane tests for the dependency-neutral runtime protocol."""

from runtime_protocol.contracts import (
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeOperationId,
    RuntimeTaskContext,
    TaskOrchestrationDelta,
)
from runtime_protocol.transport import event_from_dict, request_from_dict, result_from_dict


def test_runtime_request_round_trips_as_json_only_dto():
    request = AgentRuntimeRequest(
        run_id="run-1", thread_id="thread-1", definition_id="router_rag_agent",
        framework="langgraph", builder_id="langgraph_graph", input={"question": "hello"}, options={},
    )
    assert request_from_dict(request.to_dict()).to_dict() == request.to_dict()


def test_runtime_event_and_result_round_trip_without_framework_objects():
    event = AgentRuntimeEvent(event_id="event-1", run_id="run-1", sequence=1, kind="run.started", payload={"step": 1}, occurred_at="2026-01-01T00:00:00Z")
    result = AgentRuntimeResult(status="completed", output={"answer": "ok"}, continuation=None)
    assert event_from_dict(event.to_dict()).to_dict() == event.to_dict()
    assert result_from_dict(result.to_dict()).to_dict() == result.to_dict()


def test_task_delta_preserves_idempotency_and_opaque_continuation():
    delta = TaskOrchestrationDelta(
        event_id="event-1", attempt_id="attempt-1", operation_id="operation-1",
        idempotency_key="event-1", observed_task_version=2, observed_plan_revision=3,
    )
    value = delta.to_dict()
    assert value["idempotency_key"] == "event-1"
    assert value["observed_plan_revision"] == 3


def test_runtime_task_context_is_json_serializable():
    context = RuntimeTaskContext(task_id="task-1", objective="research", todos=[], artifact_manifests=[], permissions={}, limits={}, metadata={}, context_data={})
    assert context.to_dict()["task_id"] == "task-1"


def test_cleanup_operation_is_a_neutral_runtime_capability():
    assert RuntimeOperationId.RUN_CLEANUP.value == "run.cleanup"
