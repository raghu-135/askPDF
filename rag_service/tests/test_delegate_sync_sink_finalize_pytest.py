from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from runtime_protocol.contracts import AgentRuntimeResult

from app.product_orchestration.service import AgentRunService


@pytest.mark.asyncio
async def test_run_delegate_sync_closes_event_sink_when_child_awaits_human(monkeypatch):
    service = AgentRunService(repository=MagicMock())
    parent_run = SimpleNamespace(id="parent-run", thread_id="thread-1", user_id="user-1")
    child_run = SimpleNamespace(id="child-run", thread_id="thread-1")

    service.repository.get_run = AsyncMock(return_value=parent_run)
    service.repository.seed_builtin_workflows = AsyncMock()
    workflow = SimpleNamespace(
        id="external_research",
        spec_json={"runtime": {"chat_eligible": True}},
        framework="hermes",
        builder_id="hermes_agent",
        category="chat",
    )
    service.repository.get_workflow = AsyncMock(return_value=workflow)
    service.repository.create_run = AsyncMock(return_value=child_run)
    service.repository.update_runtime_binding = AsyncMock()
    service.repository.update_run_metadata_fields = AsyncMock()
    service.repository.mark_run_awaiting_human = AsyncMock(
        side_effect=lambda run_id, interrupt, **kwargs: SimpleNamespace(
            id=run_id,
            pending_interrupt_json=interrupt,
        )
    )
    service.repository.stamp_pending_interrupt_resume_guard = AsyncMock(
        side_effect=lambda run_id: SimpleNamespace(
            id=run_id,
            pending_interrupt_json={"interrupt_id": "int-1", "status": "pending"},
        )
    )

    monkeypatch.setattr(
        "app.product_orchestration.service.get_thread_settings",
        AsyncMock(return_value={"browser_access_mode": "ask"}),
    )
    monkeypatch.setattr(
        "app.product_orchestration.service.workflow_is_chat_eligible",
        lambda spec: True,
    )
    monkeypatch.setattr(
        "app.product_orchestration.service.definition_from_workflow",
        lambda w: SimpleNamespace(
            definition_id="external_research",
            framework="hermes",
            builder_id="hermes_agent",
        ),
    )
    monkeypatch.setattr(
        "app.product_orchestration.service.builder_for_definition",
        lambda d: MagicMock(resolve=AsyncMock(return_value={}), normalize=AsyncMock(return_value={})),
    )

    adapter = MagicMock()
    adapter.prepare_request = AsyncMock(side_effect=lambda req, **kwargs: req)
    adapter.start = AsyncMock(
        return_value=AgentRuntimeResult(
            status="awaiting_human",
            output={
                "status": "awaiting_human",
                "pending_interrupt": {"interrupt_id": "int-1", "status": "pending"},
                "answer": "",
            },
        )
    )
    monkeypatch.setattr("app.product_orchestration.service.adapter_for_definition", lambda d: adapter)
    monkeypatch.setattr(
        "app.product_orchestration.service.result_to_product_payload",
        lambda runtime_result: {
            "status": "awaiting_human",
            "pending_interrupt": {"interrupt_id": "int-1", "status": "pending"},
        },
    )
    monkeypatch.setattr(
        "app.product_orchestration.service.build_run_metrics",
        lambda result, **kwargs: {},
    )
    monkeypatch.setattr(
        "app.product_orchestration.service.AgentTraceRecorder",
        lambda run: MagicMock(
            record_interrupted_snapshot=MagicMock(),
            finalize=MagicMock(return_value={}),
        ),
    )

    close_mock = AsyncMock()
    monkeypatch.setattr("app.product_orchestration.service.close_agent_execution_sink", close_mock)

    response = await service.run_delegate_sync(
        parent_run_id="parent-run",
        thread_id="thread-1",
        child_workflow_id="external_research",
        question="Browse scholar",
        pre_fetch_bundle={},
        embedding_model="embed-model",
    )

    assert response["status"] == "awaiting_human"
    close_mock.assert_awaited_once()
    assert close_mock.await_args.kwargs["boundary_only"] is True