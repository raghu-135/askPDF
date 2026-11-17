from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from runtime_protocol.contracts import AgentRuntimeResult
from runtime_protocol.errors import RuntimeError as RuntimeContractError

from app.db import AgentRunStatus
from app.product_orchestration.delegated_parent import release_delegated_parent_if_stalled
from app.product_orchestration.service import AgentRunService


@pytest.mark.asyncio
async def test_release_delegated_parent_if_stalled_completes_running_parent():
    parent = SimpleNamespace(id="parent-run", status=AgentRunStatus.RUNNING.value, metrics_json={})
    child = SimpleNamespace(id="child-run", parent_run_id="parent-run")

    repository = MagicMock()
    repository.get_run = AsyncMock(return_value=parent)
    repository.complete_run = AsyncMock(return_value=parent)

    await release_delegated_parent_if_stalled(repository, child)

    repository.complete_run.assert_awaited_once()
    kwargs = repository.complete_run.await_args.kwargs
    assert kwargs["status"] == AgentRunStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_resume_agent_run_times_out_continue_run_and_fails_run(monkeypatch):
    service = AgentRunService(repository=MagicMock())
    service.repository._session = None
    monkeypatch.setenv("AGENT_CHAT_RESUME_DEADLINE_SECONDS", "0.05")

    resolution_run = SimpleNamespace(
        id="child-run",
        thread_id="thread-1",
        workflow_id="external_research",
        parent_run_id="parent-run",
        status=AgentRunStatus.RUNNING.value,
        task_id=None,
        metrics_json={},
        resolved_spec_json={"schema_version": 1},
        runtime_binding_json={"binding_type": "hermes.session", "payload": {}},
        pending_interrupt_json=None,
        debug_trace_json=None,
    )
    interrupt = {
        "interrupt_id": "int-1",
        "response_operation": "run.approval.respond",
        "checkpoint_resume": True,
    }

    resolution = SimpleNamespace(
        duplicate=False,
        outcome="resumed",
        interrupt=interrupt,
        run=resolution_run,
    )

    service.repository.get_run = AsyncMock(return_value=resolution_run)
    service.repository.resolve_pending_interrupt = AsyncMock(return_value=resolution)
    service.repository_factory = MagicMock(return_value=service.repository)
    service.repository.update_runtime_binding = AsyncMock()
    service.repository.update_run_metadata_fields = AsyncMock()

    async def slow_continue(*args, **kwargs):
        await asyncio.sleep(1.0)
        return AgentRuntimeResult(status="completed", output={})

    adapter = MagicMock()
    adapter.prepare_request = AsyncMock(side_effect=lambda req, **kwargs: req)
    adapter.continue_run = AsyncMock(side_effect=slow_continue)

    monkeypatch.setattr("app.product_orchestration.service.adapter_for_definition", lambda d: adapter)
    monkeypatch.setattr(
        "app.product_orchestration.service.definition_from_run",
        lambda run: SimpleNamespace(definition_id="external_research", framework="hermes", builder_id="hermes_agent"),
    )
    monkeypatch.setattr(
        "app.product_orchestration.service.resolve_run_capability_resolution",
        AsyncMock(return_value=SimpleNamespace(error=None, capabilities=SimpleNamespace(behavior={}))),
    )
    monkeypatch.setattr("app.product_orchestration.service._submit_runtime_approval", AsyncMock())

    failed_run = SimpleNamespace(
        id="child-run",
        parent_run_id="parent-run",
        status=AgentRunStatus.FAILED.value,
        metrics_json={},
    )
    lifecycle = MagicMock()
    lifecycle.complete_run = AsyncMock(return_value=failed_run)
    lifecycle.get_run = AsyncMock(return_value=failed_run)
    service.repository_factory = MagicMock(return_value=lifecycle)

    class DummySink:
        async def emit(self, *args, **kwargs):
            return None

        async def flush(self):
            return None

        async def finish(self, *args, **kwargs):
            return None

        def bind_trace_recorder(self, *args, **kwargs):
            return None

        def bind_runtime_binding_persister(self, *args, **kwargs):
            return None

        def bind_runtime_fact_persister(self, *args, **kwargs):
            return None

    sink = DummySink()

    with pytest.raises(RuntimeContractError) as excinfo:
        await service.resume_agent_run(
            "child-run",
            interrupt_id="int-1",
            action="approve",
            expected_thread_id="thread-1",
            execution_event_sink=sink,
        )
    assert excinfo.value.code == "agent_run_resume_timeout"
    lifecycle.complete_run.assert_awaited()