from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import agent_runtime_reconciliation as reconciliation
from app.services.agent_runtime_reconciliation import reconcile_known_result, result_hash
from app.services.runtime_checkpoint_reset import mark_runs_unresolved


def test_runtime_result_hash_is_stable():
    assert result_hash({"status": "completed", "output": {"answer": "ok"}}) == result_hash(
        {"output": {"answer": "ok"}, "status": "completed"}
    )


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
async def test_checkpoint_reset_preflight_marks_active_runs_unresolved(monkeypatch):
    run = SimpleNamespace(id="run-1", checkpoint_thread_id="checkpoint-1")
    updates = []

    class Repository:
        async def list_nonterminal_runtime_runs(self, *, limit):
            assert limit == 10
            return [run]

        async def update_runtime_projection(self, run_id, projection):
            updates.append((run_id, projection))

    monkeypatch.setattr("app.services.runtime_checkpoint_reset.AgentWorkflowRepository", Repository)
    result = await mark_runs_unresolved(limit=10)

    assert result == {"inspected": 1, "marked_unresolved": 0, "dry_run": 1}
    assert updates == []

    result = await mark_runs_unresolved(limit=10, dry_run=False)
    assert result["marked_unresolved"] == 1
    assert updates[0][0] == "run-1"
    assert updates[0][1]["binding_status"] == "legacy_unresolved"
