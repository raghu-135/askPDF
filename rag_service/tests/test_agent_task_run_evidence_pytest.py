from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api import agent_tasks as agent_tasks_api
from app.services.agent_task_presentation import resolve_run_evidence_manifest


def test_resolve_run_evidence_manifest_prefers_runtime_projection():
    run = SimpleNamespace(
        id="run-1",
        run_metadata_json={
            "projection": {
                "runtime_result": {
                    "task_evidence_manifest": [
                        {"id": "artifact-1", "kind": "tool_output", "sha256": "abc"},
                    ],
                    "task_incomplete_reasons": ["missing_section"],
                },
            },
        },
    )

    manifest, gaps = resolve_run_evidence_manifest(run, artifacts=[])

    assert len(manifest) == 1
    assert manifest[0]["id"] == "artifact-1"
    assert gaps == ["missing_section"]


def test_resolve_run_evidence_manifest_falls_back_to_persisted_artifacts():
    run = SimpleNamespace(id="run-1", run_metadata_json={})
    artifacts = [
        SimpleNamespace(
            id="artifact-1",
            agent_run_id="run-1",
            kind="tool_output",
            validity="valid",
            sha256="abc",
            byte_size=12,
            todo_id="todo-1",
            media_type="text/plain",
            provenance_json={"plan_revision": 1},
            source_refs_json={"tools": []},
        ),
        SimpleNamespace(
            id="artifact-2",
            agent_run_id="other-run",
            kind="tool_output",
            validity="valid",
            sha256="def",
            byte_size=8,
            todo_id=None,
            media_type="text/plain",
            provenance_json={},
            source_refs_json={},
        ),
    ]

    manifest, gaps = resolve_run_evidence_manifest(run, artifacts)

    assert len(manifest) == 1
    assert manifest[0]["id"] == "artifact-1"
    assert gaps == []


def test_resolve_run_evidence_manifest_falls_back_to_final_report_provenance():
    run = SimpleNamespace(id="run-1", run_metadata_json={})
    artifacts = [
        SimpleNamespace(
            agent_run_id="run-1",
            kind="final_report",
            validity="valid",
            provenance_json={
                "evidence_manifest": [{"id": "legacy-1", "kind": "tool_output"}],
                "evidence_gaps": ["legacy_gap"],
            },
        ),
    ]

    manifest, gaps = resolve_run_evidence_manifest(run, artifacts)

    assert manifest == [{"id": "legacy-1", "kind": "tool_output"}]
    assert gaps == ["legacy_gap"]


@pytest.mark.asyncio
async def test_run_evidence_api_returns_projection_manifest(monkeypatch):
    run = SimpleNamespace(
        id="run-1",
        task_attempt=1,
        run_metadata_json={
            "projection": {
                "runtime_result": {
                    "task_evidence_manifest": [{"id": "artifact-1", "kind": "tool_output"}],
                },
            },
        },
    )
    monkeypatch.setattr(agent_tasks_api, "_owned_task", AsyncMock())
    monkeypatch.setattr(agent_tasks_api.repository, "list_task_runs", AsyncMock(return_value=[run]))
    monkeypatch.setattr(
        agent_tasks_api.repository,
        "list_artifacts",
        AsyncMock(return_value=[]),
    )

    payload = await agent_tasks_api.get_agent_task_run_evidence("task-1", "run-1", thread_id="thread-1")

    assert payload["manifest"] == [{"id": "artifact-1", "kind": "tool_output"}]
    assert payload["evidence_gaps"] == []
