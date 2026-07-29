from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.project_lifecycle_service import (
    ProjectActiveRunsError,
    ProtectedProjectError,
)
from app.time_utils import utc_now


def _project(project_id="clone-project", name="Clone"):
    now = utc_now()
    return SimpleNamespace(
        id=project_id,
        name=name,
        description="",
        embedding_model="BAAI/bge-m3",
        settings_json={},
        created_at=now,
        updated_at=None,
    )


def test_project_lifecycle_summary_api(api_client):
    summary = {
        "project_id": "project-1",
        "thread_count": 2,
        "project_file_count": 1,
        "direct_file_count": 1,
        "unique_file_count": 2,
        "shared_file_count": 1,
        "orphan_file_count": 1,
        "memory_count": 3,
        "project_memory_count": 1,
        "thread_memory_count": 2,
        "candidate_count": 1,
        "annotation_count": 1,
        "agent_run_count": 2,
        "active_run_count": 0,
        "protected": False,
        "can_delete": True,
        "can_clone": True,
        "blocked_reason": None,
    }
    with patch(
        "app.api.projects.get_project_lifecycle_summary",
        new=AsyncMock(return_value=summary),
    ):
        response = api_client.get("/api/projects/project-1/lifecycle-summary")

    assert response.status_code == 200
    assert response.json() == summary


def test_clone_project_api_maps_project_and_counts(api_client):
    with patch(
        "app.api.projects.clone_project",
        new=AsyncMock(return_value={
            "project": _project(),
            "counts": {"threads": 2},
            "warnings": [],
        }),
    ) as clone:
        response = api_client.post(
            "/api/projects/source-project/clone",
            json={"name": "Clone", "include_threads": True},
        )

    assert response.status_code == 201
    assert response.json()["project"]["id"] == "clone-project"
    assert response.json()["counts"] == {"threads": 2}
    clone.assert_awaited_once_with(
        "source-project",
        name="Clone",
        include_threads=True,
    )


def test_clone_project_api_rejects_active_runs(api_client):
    with patch(
        "app.api.projects.clone_project",
        new=AsyncMock(side_effect=ProjectActiveRunsError("Project has active runs")),
    ):
        response = api_client.post(
            "/api/projects/source-project/clone",
            json={"name": "Clone", "include_threads": False},
        )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "active_agent_runs"


def test_delete_project_api_returns_summary(api_client):
    result = {
        "project_id": "project-1",
        "deleted": True,
        "counts": {"thread_count": 2},
        "warnings": [],
    }
    with patch(
        "app.api.projects.delete_project",
        new=AsyncMock(return_value=result),
    ):
        response = api_client.delete("/api/projects/project-1")

    assert response.status_code == 200
    assert response.json() == result


def test_delete_project_api_rejects_protected_project(api_client):
    with patch(
        "app.api.projects.delete_project",
        new=AsyncMock(side_effect=ProtectedProjectError("Protected")),
    ):
        response = api_client.delete("/api/projects/default-project")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "protected_project"
