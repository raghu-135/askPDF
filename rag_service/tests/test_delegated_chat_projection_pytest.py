from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services import agent_runtime_reconciliation as reconciliation
from app.services.agent_runtime_projection import AgentRuntimeProjection
from app.services.agent_runtime_reconciliation import result_hash


@pytest.mark.asyncio
async def test_delegated_child_updates_parent_chat_turn_instead_of_creating_duplicate(monkeypatch):
    parent_run = SimpleNamespace(
        id="parent-run",
        thread_id="thread-1",
        workflow_id="auto",
        parent_run_id=None,
        run_metadata_json={},
    )
    child_run = SimpleNamespace(
        id="child-run",
        thread_id="thread-1",
        workflow_id="external_research",
        parent_run_id="parent-run",
        run_metadata_json={},
    )
    parent_turn = SimpleNamespace(
        id="turn-parent",
        agent_run_turn_kind="assistant_final",
        agent_run_sequence=0,
        payload={"question": "Open scholar", "answer": "Waiting for approval"},
        completed_at=None,
        created_at=None,
    )
    create_calls: list[dict] = []
    update_calls: list[dict] = []

    class Repository:
        async def get_run(self, run_id):
            return child_run if run_id == "child-run" else parent_run

        async def list_chat_turns_for_run(self, run_id):
            if run_id == "child-run":
                return []
            if run_id == "parent-run":
                return [parent_turn]
            return []

        async def list_run_events(self, run_id):
            return []

        async def update_runtime_projection(self, run_id, projection):
            child_run.run_metadata_json = {"projection": dict(projection)}
            return child_run

    async def fake_create_chat_turn(**kwargs):
        create_calls.append(kwargs)
        return SimpleNamespace(
            id="turn-new",
            completed_at=None,
            created_at=None,
        )

    async def fake_update_chat_turn_result(turn_id, **kwargs):
        update_calls.append({"turn_id": turn_id, **kwargs})
        return parent_turn

    monkeypatch.setattr(reconciliation, "AgentWorkflowRepository", Repository)
    monkeypatch.setattr("app.product_orchestration.repository.AgentWorkflowRepository", Repository)
    monkeypatch.setattr("app.services.agent_runtime_projection.create_chat_turn", fake_create_chat_turn)
    monkeypatch.setattr("app.services.agent_runtime_projection.update_chat_turn_result", fake_update_chat_turn_result)
    monkeypatch.setattr("app.services.agent_runtime_projection.increment_qa_stats", AsyncMock())
    monkeypatch.setattr(
        "app.services.agent_runtime_projection._resolve_chat_memory_embedding_model",
        AsyncMock(return_value=None),
    )

    result = {
        "status": "completed",
        "question": "Open scholar",
        "final_answer": "Found https://example.com/paper",
        "answer": "Found https://example.com/paper",
    }
    projector = AgentRuntimeProjection()
    projected = await projector.project_chat_result(
        thread_id="thread-1",
        question="Open scholar",
        result=result,
        run_context={"agent_run_id": "child-run", "agent_workflow_id": "external_research"},
    )

    assert create_calls == []
    assert len(update_calls) == 1
    assert update_calls[0]["turn_id"] == "turn-parent"
    assert projected["chat_turn_id"] == "turn-parent"
    assert projected["answer"] == "Found https://example.com/paper"


@pytest.mark.asyncio
async def test_close_agent_execution_sink_finishes_delegate_awaiting_human_writer():
    from app.product_orchestration.execution_stream import AgentExecutionEventSink, close_agent_execution_sink

    sink = AgentExecutionEventSink(include_details=False)
    sink.detach_delivery()
    await sink.emit("run.started", {"run_id": "child-run"})
    await close_agent_execution_sink(sink, boundary_only=True)
    assert sink._finished is True