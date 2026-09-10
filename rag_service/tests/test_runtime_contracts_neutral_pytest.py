"""Control-plane tests for the dependency-neutral runtime protocol."""

import os
from pathlib import Path

from runtime_protocol.contracts import (
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
    RuntimeOperationId,
    RuntimeOperationOwner,
    RuntimeCapabilitySemantics,
    RuntimeSupportLevel,
    RuntimeOperationDescriptor,
    RuntimeTaskContext,
    TaskOrchestrationDelta,
)
from runtime_protocol.transport import event_from_dict, request_from_dict, result_from_dict
import pytest
from runtime_protocol.adapter import AgentRuntimeAdapter
from runtime_protocol.errors import RuntimeError


class UnsupportedAdapter(AgentRuntimeAdapter):
    framework = "test"
    builder_id = "test"

    async def capabilities(self, definition):
        raise AssertionError("No discovery should run for an unsupported default")

    async def validate(self, definition, spec, *, options=None):
        raise AssertionError("No validation should run for an unsupported default")

    async def start(self, request, *, context, event_sink=None):
        raise AssertionError("No execution should run for an unsupported default")


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "operation", "kwargs"), [
    ("get_run", "run.get", {}),
    ("wait", "run.wait", {}),
    ("stream_events", "run.events", {}),
    ("resume", "run.resume", {"interrupt": {}, "context": None}),
    ("cancel", "run.cancel", {}),
    ("pause", "task.pause", {}),
    ("respond_to_approval", "run.approval.respond", {"response": None}),
    ("send_followup", "run.send_followup", {"input": {}}),
    ("interrupt_with_input", "run.interrupt_with_input", {"input": {}}),
    ("steer_live", "run.steer_live", {"steering": None}),
    ("update_state", "run.update_state", {"input": {}}),
    ("submit_course_correction", "task.course_correction.submit", {"correction": None}),
    ("inspect_state", "run.inspect_state", {}),
    ("replay", "run.replay", {}),
    ("fork", "run.fork", {}),
    ("list_subagents", "subagent.list", {}),
    ("send_to_subagent", "subagent.send", {"subagent_id": "child", "input": {}}),
    ("cancel_subagent", "subagent.cancel", {"subagent_id": "child"}),
    ("list_artifacts", "artifact.list", {}),
])
async def test_universal_optional_operations_fail_without_execution(method, operation, kwargs):
    with pytest.raises(RuntimeError) as caught:
        await getattr(UnsupportedAdapter(), method)(None, **kwargs)
    assert caught.value.code == "runtime_capability_unsupported"
    assert caught.value.details["operation_id"] == operation


@pytest.mark.asyncio
@pytest.mark.parametrize(("method", "operation", "args", "kwargs"), [
    ("cleanup_run", "run.cleanup", ("run",), {}),
    ("list_runs", "run.list", (), {"thread_id": "thread"}),
    ("project_trace", "trace.project", ([],), {"run_id": "run"}),
])
async def test_non_request_optional_operations_are_structurally_unsupported(method, operation, args, kwargs):
    with pytest.raises(RuntimeError) as caught:
        await getattr(UnsupportedAdapter(), method)(*args, **kwargs)
    assert caught.value.code == "runtime_capability_unsupported"
    assert caught.value.details["operation_id"] == operation


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


def test_operation_descriptor_round_trips_checkpoint_boundary_requirement():
    descriptor = RuntimeOperationDescriptor(
        support=RuntimeSupportLevel.CONDITIONAL,
        owner=RuntimeOperationOwner.RUNTIME,
        enabled=True,
        semantics=RuntimeCapabilitySemantics.CHECKPOINT_STATE_INSPECTION,
        requires_checkpoint_boundary=True,
    )
    assert descriptor.to_dict()["requires_checkpoint_boundary"] is True


def test_frontend_runtime_semantics_match_backend_contract():
    repo_root = Path(os.environ.get("ASKPDF_REPO_DIR", Path(__file__).parents[2]))
    source = (repo_root / "frontend" / "src" / "lib" / "api.ts").read_text()
    for semantics in RuntimeCapabilitySemantics:
        assert f"'{semantics.value}'" in source
