from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

import app.api.agent_workflows as agent_workflows_api
import app.api.threads as threads_api
import app.agent_workflows.service as service_module
import app.runtime.cleanup as runtime_cleanup
import app.services.agent_task_repository as task_repository
import app.services.task_artifact_service as task_artifact_service
from app.agent_workflows.interrupts import InterruptResolutionResult
from app.agent_workflows.service import AgentRunService
from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeResult,
    RuntimeCapabilities,
    RuntimeOperationDescriptor,
    RuntimeOperationId,
    RuntimeOperationOwner,
    RuntimeSupportLevel,
)
from app.runtime.errors import RuntimeError
from app.runtime.registry import RuntimeRegistry
from app.services.runtime_operation_repository import RuntimeOperationConflict


class RecordingAdapter:
    framework = "fake"
    builder_id = "fake_builder"

    def __init__(self, *, unsupported=()):
        self.unsupported = {operation.value if isinstance(operation, RuntimeOperationId) else str(operation) for operation in unsupported}
        self.calls = {"approval": 0, "cancel": 0, "continue": 0, "inspect_state": 0, "resume": 0, "send_followup": 0, "steer_live": 0}
        self.approval_failures = 0
        self.deleted_continuations = 0

    async def capabilities(self, definition):
        operations = {
            RuntimeOperationId.RUN_CANCEL.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, True),
            RuntimeOperationId.RUN_INSPECT_STATE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, True),
            RuntimeOperationId.RUN_RESUME.value: RuntimeOperationDescriptor(RuntimeSupportLevel.CONDITIONAL, RuntimeOperationOwner.RUNTIME, True),
            RuntimeOperationId.RUN_APPROVAL_RESPOND.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, True),
            RuntimeOperationId.RUN_SEND_FOLLOWUP.value: RuntimeOperationDescriptor(RuntimeSupportLevel.NATIVE, RuntimeOperationOwner.RUNTIME, True),
            RuntimeOperationId.RUN_STEER_LIVE.value: RuntimeOperationDescriptor(RuntimeSupportLevel.UNSUPPORTED, RuntimeOperationOwner.RUNTIME, False, disabled_reason="runtime_capability_unsupported"),
            RuntimeOperationId.RUN_CONTINUATION_CLEANUP.value: RuntimeOperationDescriptor(RuntimeSupportLevel.UNSUPPORTED, RuntimeOperationOwner.RUNTIME, False, disabled_reason="runtime_capability_unsupported"),
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

    async def inspect_state(self, request):
        self.calls["inspect_state"] += 1
        return {"state": {}}

    async def resume(self, request, *, interrupt, context, event_sink=None):
        self.calls["resume"] += 1
        return None

    async def respond_to_approval(self, request, response):
        self.calls["approval"] += 1
        if self.approval_failures:
            self.approval_failures -= 1
            raise RuntimeError("runtime_approval_failed", "Approval submission failed")
        return {"status": "accepted"}

    async def continue_run(self, request, *, context, event_sink=None):
        self.calls["continue"] += 1
        return AgentRuntimeResult(status="completed", output="done")

    async def delete_continuation(self, continuation):
        self.deleted_continuations += 1
        return {"status": "deleted"}

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
        self.restored = 0

    async def get_run(self, run_id):
        return self.run if run_id == self.run.id else None

    async def resolve_pending_interrupt(self, *args, **kwargs):
        return self.resolution

    async def complete_run(self, run_id, **kwargs):
        self.run.status = kwargs.get("status", self.run.status)
        return self.run

    async def restore_pending_approval_after_runtime_failure(self, *args, **kwargs):
        self.restored += 1
        return True

    async def mark_run_awaiting_human(self, *args, **kwargs):
        return self.run

    async def set_run_debug_trace(self, run_id, payload):
        self.run.debug_trace_json = payload
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
        debug_trace_json=None,
        checkpoint_thread_id="run-1",
        completed_at=None,
    )


def _patch_runtime(monkeypatch, adapter, repository):
    registry = RuntimeRegistry(adapters=[adapter])
    monkeypatch.setattr(service_module, "get_runtime_registry", lambda: registry)
    monkeypatch.setattr(service_module, "adapter_for_definition", lambda definition: adapter)
    monkeypatch.setattr(
        service_module,
        "claim_runtime_operation",
        AsyncMock(return_value=SimpleNamespace(id="operation-1", status="in_progress", result_json={}, error_json=None)),
    )
    monkeypatch.setattr(service_module, "complete_runtime_operation", AsyncMock())
    monkeypatch.setattr(service_module, "fail_runtime_operation", AsyncMock())
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
async def test_approval_continuation_is_gated_before_approval_or_continue(monkeypatch):
    pending = {
        "interrupt_id": "approval-unsupported",
        "status": "pending",
        "response_operation": RuntimeOperationId.RUN_APPROVAL_RESPOND.value,
        "checkpoint_resume": True,
    }
    run = _run(status="awaiting_human", pending=pending)
    adapter = RecordingAdapter(unsupported={RuntimeOperationId.RUN_APPROVAL_RESPOND})
    service = _patch_runtime(monkeypatch, adapter, FakeRepository(run))

    with pytest.raises(RuntimeError) as caught:
        await service.resume_agent_run(
            run.id,
            interrupt_id="approval-unsupported",
            action="approve",
            expected_thread_id=run.thread_id,
            execution_event_sink=Sink(),
        )

    assert caught.value.code == "runtime_capability_unsupported"
    assert caught.value.details["operation_id"] == RuntimeOperationId.RUN_APPROVAL_RESPOND.value
    assert adapter.calls["approval"] == 0
    assert adapter.calls["continue"] == 0


@pytest.mark.asyncio
async def test_task_runtime_approval_failure_can_be_retried(monkeypatch):
    pending = {
        "interrupt_id": "approval-1",
        "status": "pending",
        "response_operation": RuntimeOperationId.RUN_APPROVAL_RESPOND.value,
        "checkpoint_resume": True,
    }
    current = _run(status="awaiting_human", pending=pending)
    resolved_interrupt = {**pending, "status": "resumed", "decision": {"action": "approve"}}
    resolved_run = _run(status="running", pending=resolved_interrupt)
    resolved_run.task_id = "task-1"
    resolution = InterruptResolutionResult(run=resolved_run, outcome="resumed", interrupt=resolved_interrupt)
    repository = FakeRepository(current, resolution)
    adapter = RecordingAdapter()
    adapter.approval_failures = 1
    service = _patch_runtime(monkeypatch, adapter, repository)
    queue = AsyncMock()
    monkeypatch.setattr("app.services.agent_task_repository.queue_task_after_interrupt", queue)

    with pytest.raises(RuntimeError, match="Approval submission failed"):
        await service.resume_agent_run(
            current.id,
            interrupt_id="approval-1",
            action="approve",
            expected_thread_id=current.thread_id,
            execution_event_sink=Sink(),
        )

    assert repository.restored == 1
    queue.assert_not_awaited()

    await service.resume_agent_run(
        current.id,
        interrupt_id="approval-1",
        action="approve",
        expected_thread_id=current.thread_id,
        execution_event_sink=Sink(),
    )

    assert adapter.calls["approval"] == 2
    assert adapter.calls["resume"] == 0
    queue.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_task_runtime_approval_responds_then_continues_without_resume(monkeypatch):
    pending = {
        "interrupt_id": "approval-2",
        "status": "pending",
        "response_operation": RuntimeOperationId.RUN_APPROVAL_RESPOND.value,
        "checkpoint_resume": True,
    }
    current = _run(status="awaiting_human", pending=pending)
    resolved_interrupt = {**pending, "status": "resumed", "decision": {"action": "approve"}}
    resolved_run = _run(status="running", pending=resolved_interrupt)
    resolution = InterruptResolutionResult(run=resolved_run, outcome="resumed", interrupt=resolved_interrupt)
    repository = FakeRepository(current, resolution)
    adapter = RecordingAdapter()
    service = _patch_runtime(monkeypatch, adapter, repository)
    service.projection.project_chat_result = AsyncMock(return_value={"status": "completed", "answer": "done"})

    result = await service.resume_agent_run(
        current.id,
        interrupt_id="approval-2",
        action="approve",
        expected_thread_id=current.thread_id,
        execution_event_sink=Sink(),
    )

    assert result.run.status == "completed"
    assert adapter.calls["approval"] == 1
    assert adapter.calls["continue"] == 1
    assert adapter.calls["resume"] == 0


def _hermes_bound_run():
    run = _run(status="completed")
    run.framework = "hermes"
    run.builder_id = "hermes_agent"
    run.runtime_binding_json = {
        "binding_type": "hermes_session",
        "payload": {"session_id": "session-1", "upstream_run_id": "upstream-1"},
    }
    return run


@pytest.mark.asyncio
async def test_hermes_task_deletion_skips_unsupported_session_cleanup(monkeypatch):
    adapter = RecordingAdapter()
    adapter.framework = "hermes"
    adapter.builder_id = "hermes_agent"
    run = _hermes_bound_run()
    monkeypatch.setattr(runtime_cleanup, "adapter_for_definition", lambda definition: adapter)
    monkeypatch.setattr(task_artifact_service, "get_content_store", lambda: SimpleNamespace(delete=AsyncMock()))
    monkeypatch.setattr(task_repository, "list_artifacts", AsyncMock(return_value=[]))
    monkeypatch.setattr(task_repository, "list_task_runs", AsyncMock(return_value=[run]))
    completed = AsyncMock()
    monkeypatch.setattr(task_repository, "mark_task_deletion_completed", completed)

    await task_artifact_service.cleanup_deleted_task("task-1")

    completed.assert_awaited_once_with("task-1")
    assert adapter.deleted_continuations == 0


@pytest.mark.asyncio
async def test_hermes_thread_deletion_continues_with_existing_session_binding(monkeypatch):
    adapter = RecordingAdapter()
    adapter.framework = "hermes"
    adapter.builder_id = "hermes_agent"
    run = _hermes_bound_run()
    monkeypatch.setattr(runtime_cleanup, "adapter_for_definition", lambda definition: adapter)

    async def delete_task_resources(thread_ids):
        assert thread_ids == ["thread-1"]
        await runtime_cleanup.delete_run_continuation(run)

    monkeypatch.setattr(task_artifact_service, "delete_task_resources_for_threads", delete_task_resources)
    monkeypatch.setattr(threads_api, "get_thread", AsyncMock(return_value=SimpleNamespace(id="thread-1", embedding_model="embed")))
    monkeypatch.setattr(threads_api, "get_thread_files", AsyncMock(return_value=[]))
    monkeypatch.setattr(threads_api, "get_vector_db", lambda: SimpleNamespace(delete_thread_data=AsyncMock(return_value=True)))
    monkeypatch.setattr(threads_api, "hard_delete_thread_memory_resources", AsyncMock(return_value={}))
    monkeypatch.setattr("app.services.embedding_materialization_service.cancel_embedding_jobs_for_scope", AsyncMock())
    monkeypatch.setattr(threads_api, "delete_thread", AsyncMock(return_value=True))

    assert await threads_api._delete_thread_resources("thread-1") is True
    assert adapter.deleted_continuations == 0


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


@pytest.mark.asyncio
async def test_runtime_operation_idempotency_replays_completed_result_without_adapter_call(monkeypatch):
    adapter = RecordingAdapter()
    run = _run()
    service = _patch_runtime(monkeypatch, adapter, FakeRepository(run))
    record = SimpleNamespace(
        id="operation-1",
        status="completed",
        result_json={"status": "queued", "sequence": 1},
        error_json=None,
    )
    monkeypatch.setattr(service_module, "claim_runtime_operation", AsyncMock(return_value=record))

    result = await service.operate_agent_run(
        run,
        RuntimeOperationId.RUN_SEND_FOLLOWUP,
        input={"text": "follow up"},
        idempotency_key="same-request",
    )

    assert result == record.result_json
    assert adapter.calls["send_followup"] == 0


@pytest.mark.asyncio
async def test_runtime_operation_idempotency_conflict_prevents_adapter_call(monkeypatch):
    adapter = RecordingAdapter()
    run = _run()
    service = _patch_runtime(monkeypatch, adapter, FakeRepository(run))
    monkeypatch.setattr(
        service_module,
        "claim_runtime_operation",
        AsyncMock(side_effect=RuntimeOperationConflict("runtime_operation_idempotency_conflict", "key reused")),
    )

    with pytest.raises(RuntimeError) as caught:
        await service.operate_agent_run(
            run,
            RuntimeOperationId.RUN_SEND_FOLLOWUP,
            input={"text": "different"},
            idempotency_key="same-request",
        )

    assert caught.value.code == "runtime_operation_idempotency_conflict"
    assert adapter.calls["send_followup"] == 0


@pytest.mark.asyncio
async def test_runtime_operation_in_progress_duplicate_prevents_adapter_call(monkeypatch):
    adapter = RecordingAdapter()
    run = _run()
    service = _patch_runtime(monkeypatch, adapter, FakeRepository(run))
    existing = SimpleNamespace(status="in_progress")
    monkeypatch.setattr(
        service_module,
        "claim_runtime_operation",
        AsyncMock(side_effect=RuntimeOperationConflict("runtime_operation_in_progress", "already running", operation=existing)),
    )

    with pytest.raises(RuntimeError) as caught:
        await service.operate_agent_run(
            run,
            RuntimeOperationId.RUN_SEND_FOLLOWUP,
            input={"text": "duplicate"},
            idempotency_key="in-flight",
        )

    assert caught.value.code == "runtime_operation_in_progress"
    assert adapter.calls["send_followup"] == 0


@pytest.mark.asyncio
async def test_runtime_operation_adapter_failure_is_persisted(monkeypatch):
    adapter = RecordingAdapter()
    adapter.send_followup = AsyncMock(side_effect=RuntimeError("adapter_failed", "Adapter failed"))
    run = _run()
    service = _patch_runtime(monkeypatch, adapter, FakeRepository(run))
    fail = service_module.fail_runtime_operation

    with pytest.raises(RuntimeError) as caught:
        await service.operate_agent_run(
            run,
            RuntimeOperationId.RUN_SEND_FOLLOWUP,
            input={"text": "once"},
            idempotency_key="failed-request",
        )

    assert caught.value.code == "adapter_failed"
    fail.assert_awaited_once()
    assert adapter.send_followup.await_count == 1


@pytest.mark.asyncio
async def test_retryable_capability_failure_can_retry_same_idempotency_key(monkeypatch):
    class FlakyCapabilityAdapter(RecordingAdapter):
        def __init__(self):
            super().__init__()
            self.fail_capability_discovery = True

        async def capabilities(self, definition):
            if self.fail_capability_discovery:
                raise RuntimeError(
                    "runtime_capability_discovery_failed",
                    "Runtime capability discovery failed",
                    retryable=True,
                )
            return await super().capabilities(definition)

    adapter = FlakyCapabilityAdapter()
    run = _run()
    service = _patch_runtime(monkeypatch, adapter, FakeRepository(run))
    record = SimpleNamespace(id="operation-1", status="in_progress", result_json={}, error_json=None)
    monkeypatch.setattr(service_module, "claim_runtime_operation", AsyncMock(return_value=record))
    fail = service_module.fail_runtime_operation

    with pytest.raises(RuntimeError) as first:
        await service.operate_agent_run(
            run,
            RuntimeOperationId.RUN_SEND_FOLLOWUP,
            input={"text": "retry me"},
            idempotency_key="retryable-capability",
        )
    assert first.value.retryable is True
    fail.assert_awaited_once()

    adapter.fail_capability_discovery = False
    result = await service.operate_agent_run(
        run,
        RuntimeOperationId.RUN_SEND_FOLLOWUP,
        input={"text": "retry me"},
        idempotency_key="retryable-capability",
    )

    assert result == {"status": "queued"}
    assert adapter.calls["send_followup"] == 1
