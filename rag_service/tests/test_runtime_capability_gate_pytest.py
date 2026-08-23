from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import app.api.agent_workflows as agent_workflows_api
import app.agent_workflows.service as service_module
from app.agent_workflows.interrupts import InterruptResolutionResult
from app.agent_workflows.service import AgentRunService
from app.runtime.contracts import (
    AgentDefinition,
    RuntimeCapabilities,
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeSupportLevel,
)
from app.runtime.errors import RuntimeError
from app.runtime.registry import RuntimeRegistry


class RecordingAdapter:
    framework = "fake"
    builder_id = "fake_builder"

    def __init__(self, *, unsupported=()):
        self.unsupported = {operation.value if isinstance(operation, RuntimeOperationId) else str(operation) for operation in unsupported}
        self.calls = {"cancel": 0, "inspect_state": 0, "resume": 0, "send_followup": 0, "steer_live": 0}

    async def capabilities(self, definition):
        operations = {
            RuntimeOperationId.RUN_CANCEL.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, True),
            RuntimeOperationId.RUN_INSPECT_STATE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, True),
            RuntimeOperationId.RUN_RESUME.value: RuntimeOperationDescriptor(RuntimeSupportLevel.CONDITIONAL, True),
            RuntimeOperationId.RUN_SEND_FOLLOWUP.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, True),
            RuntimeOperationId.RUN_STEER_LIVE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.UNSUPPORTED, False, disabled_reason="runtime_capability_unsupported"),
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

    async def inspect_state(self, request):
        self.calls["inspect_state"] += 1
        return {"state": {}}

    async def resume(self, request, *, interrupt, context, event_sink=None):
        self.calls["resume"] += 1
        return None

    async def send_followup(self, request, input):
        self.calls["send_followup"] += 1
        return {"status": "queued"}

    async def steer_live(self, request, steering):
        self.calls["steer_live"] += 1
        return {"status": "steered"}


class FakeRepository:
    def __init__(self, run, resolution=None):
        self.run = run
        self.resolution = resolution

    async def get_run(self, run_id):
        return self.run if run_id == self.run.id else None

    async def resolve_pending_interrupt(self, *args, **kwargs):
        return self.resolution

    async def complete_run(self, run_id, **kwargs):
        return self.run


class Sink:
    async def emit(self, *args, **kwargs):
        return None


def _run(*, status="running", pending=None):
    return SimpleNamespace(
        id="run-1",
        thread_id="thread-1",
        workflow_id="definition-1",
        framework="fake",
        builder_id="fake_builder",
        definition_category=None,
        resolved_spec_json={},
        runtime_binding_json={"binding_type": "fake"},
        runtime_binding_status="active",
        status=status,
        pending_interrupt_json=pending,
        task_id=None,
        metrics_json={},
    )


def _patch_runtime(monkeypatch, adapter, repository):
    registry = RuntimeRegistry(adapters=[adapter])
    monkeypatch.setattr(service_module, "get_runtime_registry", lambda: registry)
    monkeypatch.setattr(service_module, "adapter_for_definition", lambda definition: adapter)
    return AgentRunService(repository=repository)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "call_name", "operation"),
    [
        ("cancel_agent_run", "cancel", RuntimeOperationId.RUN_CANCEL),
        ("inspect_agent_run", "inspect_state", RuntimeOperationId.RUN_INSPECT_STATE),
    ],
)
async def test_direct_service_operations_reject_before_adapter_invocation(monkeypatch, method, call_name, operation):
    adapter = RecordingAdapter(unsupported={operation})
    run = _run()
    service = _patch_runtime(monkeypatch, adapter, FakeRepository(run))

    with pytest.raises(RuntimeError) as caught:
        if method == "cancel_agent_run":
            await service.cancel_agent_run(run.id, thread_id=run.thread_id)
        else:
            await service.inspect_agent_run(run)

    assert caught.value.code == "runtime_capability_unsupported"
    assert caught.value.details["operation_id"] == operation.value
    assert adapter.calls[call_name] == 0


@pytest.mark.asyncio
async def test_resume_rejects_before_runtime_resume_invocation(monkeypatch):
    interrupt = {
        "interrupt_id": "interrupt-1",
        "status": "pending",
        "response_operation": RuntimeOperationId.RUN_RESUME.value,
        "checkpoint_resume": True,
    }
    run = _run(status="awaiting_human", pending=interrupt)
    resolution = InterruptResolutionResult(run=run, outcome="resumed", interrupt=interrupt)
    adapter = RecordingAdapter(unsupported={RuntimeOperationId.RUN_RESUME})
    service = _patch_runtime(monkeypatch, adapter, FakeRepository(run, resolution))

    with pytest.raises(RuntimeError) as caught:
        await service.resume_agent_run(
            run.id,
            interrupt_id="interrupt-1",
            action="approve",
            expected_thread_id=run.thread_id,
            execution_event_sink=Sink(),
        )

    assert caught.value.code == "runtime_capability_unsupported"
    assert caught.value.details["operation_id"] == RuntimeOperationId.RUN_RESUME.value
    assert adapter.calls["resume"] == 0


@pytest.mark.asyncio
async def test_supported_followup_reaches_adapter_once(monkeypatch):
    run = _run()
    adapter = RecordingAdapter()
    service = _patch_runtime(monkeypatch, adapter, FakeRepository(run))

    result = await service.operate_agent_run(
        run,
        RuntimeOperationId.RUN_SEND_FOLLOWUP,
        input={"text": "continue"},
        idempotency_key="request-1",
    )

    assert result == {"status": "queued"}
    assert adapter.calls["send_followup"] == 1


@pytest.mark.asyncio
async def test_http_operation_boundary_preserves_structured_capability_rejection(monkeypatch):
    run = _run()
    adapter = RecordingAdapter(unsupported={RuntimeOperationId.RUN_STEER_LIVE})
    service = _patch_runtime(monkeypatch, adapter, FakeRepository(run))

    async def owned_run(*args, **kwargs):
        return run

    monkeypatch.setattr(agent_workflows_api, "_owned_run_for_operation", owned_run)
    monkeypatch.setattr(agent_workflows_api, "AgentRunService", lambda: service)

    with pytest.raises(HTTPException) as caught:
        await agent_workflows_api._execute_run_operation(
            run.id,
            RuntimeOperationId.RUN_STEER_LIVE,
            thread_id=run.thread_id,
            input={"text": "guide"},
            idempotency_key="request-2",
        )

    assert caught.value.status_code == 409
    detail = caught.value.detail
    assert detail["code"] == "runtime_capability_unsupported"
    assert detail["details"]["operation_id"] == RuntimeOperationId.RUN_STEER_LIVE.value
    assert detail["details"]["framework"] == adapter.framework
    assert detail["details"]["builder_id"] == adapter.builder_id
    assert detail["details"]["support_level"] == "unsupported"
    assert adapter.calls["steer_live"] == 0


@pytest.mark.asyncio
async def test_cancel_http_boundary_returns_structured_capability_rejection(monkeypatch):
    async def ready_thread(thread_id):
        return SimpleNamespace(id=thread_id)

    async def rejected_cancel(run_id, *, thread_id):
        raise RuntimeError.capability_unsupported(
            operation_id=RuntimeOperationId.RUN_CANCEL.value,
            framework="fake",
            builder_id="fake_builder",
            explanation="cancel disabled for this deployment",
        )

    monkeypatch.setattr(agent_workflows_api, "get_thread", ready_thread)
    monkeypatch.setattr(agent_workflows_api, "request_chat_run_cancel", rejected_cancel)

    with pytest.raises(HTTPException) as caught:
        await agent_workflows_api.cancel_chat_agent_run(
            "run-1",
            agent_workflows_api.AgentRunCancelRequest(thread_id="thread-1"),
        )

    assert caught.value.status_code == 409
    assert caught.value.detail["code"] == "runtime_capability_unsupported"
    assert caught.value.detail["details"]["operation_id"] == RuntimeOperationId.RUN_CANCEL.value


@pytest.mark.asyncio
async def test_chat_cancel_normalizes_http_adapter_mapping(monkeypatch):
    class FakeService:
        async def cancel_agent_run(self, run_id, *, thread_id):
            assert run_id == "run-1"
            assert thread_id == "thread-1"
            return {"status": "cancel_requested", "run_id": run_id, "run_status": "running"}

    monkeypatch.setattr(agent_workflows_api, "AgentRunService", FakeService)

    result = await agent_workflows_api.request_chat_run_cancel("run-1", thread_id="thread-1")

    assert result.status == "cancel_requested"
    assert result.run_id == "run-1"
    assert result.run_status == "running"


@pytest.mark.asyncio
async def test_resume_http_boundary_returns_structured_capability_rejection(monkeypatch):
    async def ready_thread(thread_id):
        return SimpleNamespace(id=thread_id)

    class RejectedService:
        async def resume_agent_run(self, *args, **kwargs):
            raise RuntimeError.capability_unsupported(
                operation_id=RuntimeOperationId.RUN_RESUME.value,
                framework="fake",
                builder_id="fake_builder",
                explanation="resume requires durable state",
            )

    monkeypatch.setattr(agent_workflows_api, "_require_ready_thread", ready_thread)
    monkeypatch.setattr(agent_workflows_api, "AgentRunService", RejectedService)
    request = agent_workflows_api.AgentRunResumeRequest(
        thread_id="thread-1",
        interrupt_id="interrupt-1",
        action="approve",
    )

    with pytest.raises(HTTPException) as caught:
        await agent_workflows_api.resume_agent_run("run-1", request)

    assert caught.value.status_code == 409
    assert caught.value.detail["code"] == "runtime_capability_unsupported"
    assert caught.value.detail["details"]["operation_id"] == RuntimeOperationId.RUN_RESUME.value
