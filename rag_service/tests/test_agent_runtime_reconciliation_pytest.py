from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import agent_runtime_reconciliation as reconciliation
from app.services.agent_runtime_reconciliation import reconcile_known_result, result_hash
from app.services.agent_runtime_projection import AgentRuntimeProjection
from app.services.runtime_checkpoint_reset import mark_runs_deferred


def test_runtime_result_hash_is_stable():
    assert result_hash({"status": "completed", "output": {"answer": "ok"}}) == result_hash(
        {"output": {"answer": "ok"}, "status": "completed"}
    )


@pytest.mark.asyncio
async def test_terminal_projection_replay_does_not_repeat_product_side_effects(monkeypatch):
    run = SimpleNamespace(
        id="run-1",
        thread_id="thread-1",
        workflow_id="workflow-1",
        run_metadata_json={},
    )
    persisted_run = SimpleNamespace(id=run.id, run_metadata_json={})
    turns = []
    projection_updates = []
    index_calls = []
    compact_calls = []
    stats_calls = []

    class Repository:
        async def get_run(self, run_id):
            assert run_id == run.id
            return persisted_run

        async def list_chat_turns_for_run(self, run_id):
            assert run_id == run.id
            return list(turns)

        async def list_run_events(self, run_id):
            assert run_id == run.id
            return []

        async def update_runtime_projection(self, run_id, projection):
            assert run_id == run.id
            projection_updates.append(dict(projection))
            persisted_run.run_metadata_json = {"projection": dict(projection)}
            return persisted_run

    async def fake_create_chat_turn(**kwargs):
        turn = SimpleNamespace(
            id="turn-1",
            completed_at=None,
            created_at=None,
            agent_run_turn_kind=kwargs["agent_run_turn_kind"],
            agent_run_sequence=kwargs["agent_run_sequence"],
        )
        turns.append(turn)
        return turn

    async def fake_index_chat_memory_for_thread(**kwargs):
        index_calls.append(kwargs)
        return {"memory_compact_text": "compact"}

    async def fake_update_message_context_compact(turn_id, compact):
        compact_calls.append((turn_id, compact))

    async def fake_increment_qa_stats(thread_id, qa_chars):
        stats_calls.append((thread_id, qa_chars))

    monkeypatch.setattr(reconciliation, "AgentWorkflowRepository", Repository)
    monkeypatch.setattr("app.agent_workflows.repository.AgentWorkflowRepository", Repository)
    monkeypatch.setattr("app.services.agent_runtime_projection.create_chat_turn", fake_create_chat_turn)
    monkeypatch.setattr("app.services.agent_runtime_projection.index_chat_memory_for_thread", fake_index_chat_memory_for_thread)
    monkeypatch.setattr("app.services.agent_runtime_projection.update_message_context_compact", fake_update_message_context_compact)
    monkeypatch.setattr("app.services.agent_runtime_projection.increment_qa_stats", fake_increment_qa_stats)

    result = {
        "status": "completed",
        "question": "Question?",
        "answer": "Answer.",
        "embedding_model": "embed",
        "llm_model": "llm",
    }
    projector = AgentRuntimeProjection()

    first = await projector.project_terminal_result(run=run, result=result, terminal_event_id="event-1")
    second = await projector.project_terminal_result(run=run, result=result, terminal_event_id="event-1")

    assert first["chat_turn_id"] == second["chat_turn_id"] == "turn-1"
    assert len(turns) == 1
    assert len(index_calls) == 1
    assert compact_calls == [("turn-1", "compact")]
    assert stats_calls == [("thread-1", len("Question?") + len("Answer."))]
    assert projection_updates[-1]["status"] == "applied"
    assert projection_updates[-1]["result_hash"] == result_hash(result)


@pytest.mark.asyncio
async def test_reconciliation_preserves_paused_runs():
    run = SimpleNamespace(id="run-1", status="awaiting_human")

    class Projector:
        async def reconcile_run(self, **_kwargs):
            raise AssertionError("paused runs must not be projected")

    result = await reconcile_known_result(
        run,
        {"status": "awaiting_human", "pending_interrupt": {"interrupt_id": "i-1"}},
        Projector(),
    )
    assert result is run


@pytest.mark.asyncio
async def test_reconciliation_does_not_project_unknown_result():
    run = SimpleNamespace(id="run-1", status="running")

    class Projector:
        async def reconcile_run(self, **_kwargs):
            raise AssertionError("unknown outcomes must remain unresolved")

    result = await reconcile_known_result(run, None, Projector())
    assert result is run


@pytest.mark.asyncio
async def test_reconcile_run_by_id_projects_persisted_terminal_result(monkeypatch):
    run = SimpleNamespace(
        id="run-1",
        thread_id="thread-1",
        workflow_id="workflow-1",
        framework="fake",
        builder_id="fake_builder",
        definition_category=None,
        task_id=None,
        status="running",
        resolved_spec_json={},
        runtime_binding_json={"binding_type": "fake", "payload": {}},
        runtime_binding_status="active",
        run_metadata_json={
            "projection": {
                "runtime_result": {"status": "completed", "answer": "durable answer"},
            },
        },
    )
    persisted = SimpleNamespace(**vars(run))
    updates = []
    projected = []

    class Repository:
        async def get_run(self, run_id):
            assert run_id == run.id
            return persisted

        async def update_runtime_projection(self, run_id, projection):
            assert run_id == run.id
            updates.append(dict(projection))
            persisted.run_metadata_json = {"projection": dict(projection)}
            return persisted

    class Adapter:
        async def inspect_state(self, request):
            raise AssertionError("known terminal results do not require inspection")

    class Projector:
        async def reconcile_run(self, **kwargs):
            projected.append(kwargs)
            return persisted

    monkeypatch.setattr("app.agent_workflows.repository.AgentWorkflowRepository", Repository)
    monkeypatch.setattr(reconciliation, "AgentWorkflowRepository", Repository)
    monkeypatch.setattr("app.runtime.registry.get_runtime_registry", lambda: SimpleNamespace(get=lambda definition: Adapter()))
    monkeypatch.setattr("app.services.agent_runtime_projection.AgentRuntimeProjection", Projector)

    result = await reconciliation.reconcile_run_by_id(run.id)

    assert result == "projected"
    assert projected and projected[0]["result"]["answer"] == "durable answer"
    assert updates[-1]["reconciliation_status"] == "projected"


@pytest.mark.asyncio
async def test_bounded_reconciliation_reports_candidate_outcomes(monkeypatch):
    runs = [SimpleNamespace(id="run-1"), SimpleNamespace(id="run-2"), SimpleNamespace(id="run-3")]

    class Repository:
        async def list_runtime_reconciliation_candidates(self, *, limit):
            assert limit == 3
            return runs

    statuses = {"run-1": "projected", "run-2": "preserved", "run-3": "deferred"}

    async def reconcile(run_id, *, dry_run=False):
        assert dry_run is False
        return statuses[run_id]

    monkeypatch.setattr(reconciliation.AgentWorkflowRepository, "list_runtime_reconciliation_candidates", Repository().list_runtime_reconciliation_candidates)
    monkeypatch.setattr(reconciliation, "reconcile_run_by_id", reconcile)

    assert await reconciliation.run_runtime_reconciliation(batch_size=3) == {
        "inspected": 3,
        "projected": 1,
        "preserved": 1,
        "failed": 0,
        "deferred": 1,
    }


@pytest.mark.asyncio
async def test_checkpoint_reset_preflight_marks_active_runs_deferred(monkeypatch):
    run = SimpleNamespace(
        id="run-1",
        runtime_binding_json={
            "binding_type": "langgraph.checkpoint",
            "payload": {"checkpoint_thread_id": "checkpoint-1"},
        },
        runtime_binding_status="active",
    )
    updates = []

    class Repository:
        async def list_nonterminal_runtime_runs(self, *, limit):
            assert limit == 10
            return [run]

        async def update_runtime_projection(self, run_id, projection):
            updates.append((run_id, projection))

    monkeypatch.setattr("app.services.runtime_checkpoint_reset.AgentWorkflowRepository", Repository)
    result = await mark_runs_deferred(limit=10)

    assert result == {"inspected": 1, "marked_deferred": 0, "dry_run": 1}
    assert updates == []

    result = await mark_runs_deferred(limit=10, dry_run=False)
    assert result["marked_deferred"] == 1
    assert updates[0][0] == "run-1"
    assert updates[0][1]["reconciliation_status"] == "deferred"
    assert "binding_status" not in updates[0][1]
