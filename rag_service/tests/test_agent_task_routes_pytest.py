from app.api import agent_tasks
from fastapi import HTTPException
import pytest
from starlette.routing import Match


def _resolved_endpoint(path: str) -> str | None:
    scope = {"type": "http", "method": "POST", "path": path, "root_path": "", "app": None}
    for route in agent_tasks.router.routes:
        match, _ = route.matches(scope)
        if match == Match.FULL:
            return route.endpoint.__name__
    return None


def test_agent_task_post_routes_do_not_shadow_static_actions():
    post_paths = {
        route.path for route in agent_tasks.router.routes
        if "POST" in getattr(route, "methods", set())
    }

    assert "/agent-tasks/{task_id}/commands/{action}" in post_paths
    assert "/agent-tasks/{task_id}/{action}" not in post_paths
    assert "/agent-tasks/{task_id}/course-corrections" in post_paths
    assert "/agent-tasks/{task_id}/result-review/responses" in post_paths
    assert "/agent-tasks/{task_id}/budget-review/responses" in post_paths

    assert _resolved_endpoint("/agent-tasks/task-1/course-corrections") == "submit_agent_task_course_correction"
    assert _resolved_endpoint("/agent-tasks/task-1/result-review/responses") == "respond_to_agent_task_result_review"
    assert _resolved_endpoint("/agent-tasks/task-1/budget-review/responses") == "respond_to_agent_task_budget_review"
    assert _resolved_endpoint("/agent-tasks/task-1/commands/not-an-action") == "command_agent_task"
    assert _resolved_endpoint("/agent-tasks/task-1/not-an-action") is None


@pytest.mark.asyncio
async def test_unknown_lifecycle_action_is_scoped_to_commands_route():
    with pytest.raises(HTTPException) as error:
        await agent_tasks.command_agent_task(
            "task-1",
            "not-an-action",
            agent_tasks.AgentTaskCommandRequest(expected_version=1),
            thread_id="thread-1",
            idempotency_key="unknown-action",
        )

    assert error.value.status_code == 404
    assert error.value.detail == {"code": "task_command_unknown"}
