from types import SimpleNamespace

import pytest

from app.api import agent_tasks
from runtime_protocol.contracts import RuntimeCourseCorrectionReceipt


def _task():
    return SimpleNamespace(
        id="task-1",
        thread_id="thread-1",
        project_id="project-1",
        workflow_id="deep-research",
        objective="Research thoroughly",
        status="queued",
        version=4,
        primary_run_id="run-1",
        active_run_id="run-1",
        latest_run_attempt=1,
        progress=25,
        completed_todos=1,
        total_todos=4,
        current_phase="budget_correction_delivery_pending",
        terminal_reason=None,
        budgets_json={},
        config_json={},
        created_at=None,
        updated_at=None,
        started_at=None,
        paused_at=None,
        completed_at=None,
        expires_at=None,
    )


@pytest.mark.asyncio
async def test_budget_steer_delivers_durable_command_before_releasing_resume(monkeypatch):
    task = _task()
    run = SimpleNamespace(
        id="run-1",
        thread_id="thread-1",
        framework="langgraph",
        builder_id="langgraph_graph",
    )
    command = SimpleNamespace(
        id="command-1",
        status="accepted",
        result_json={
            "delivery_state": "accepted",
            "correction": {
                "correction_id": "correction-1",
                "instruction": "Investigate the remaining evidence.",
                "scope": "remaining_work",
                "observed_task_version": 4,
                "observed_plan_revision": 2,
            },
        },
    )
    submitted = []
    delivered = []

    async def owned_task(*_args, **_kwargs):
        return task

    async def get_task_run(_task_id):
        return run

    async def require_capability(*_args, **_kwargs):
        return None

    async def respond_to_budget_review(*_args, **_kwargs):
        return task, False, False

    async def get_command(*_args, **_kwargs):
        return command

    async def get_task(_task_id):
        return task

    async def mark_delivered(command_id, *, receipt):
        delivered.append((command_id, receipt))
        task.current_phase = "budget_continuation_queued"

    class Adapter:
        async def submit_course_correction(self, request, correction):
            submitted.append((request, correction))
            return RuntimeCourseCorrectionReceipt(
                correction_id=correction.correction_id,
                operation_id=correction.operation_id,
                status="accepted",
                run_id=request.run_id,
                run_status="awaiting_human",
            )

    definition = SimpleNamespace(
        definition_id="deep-research",
        framework="langgraph",
        builder_id="langgraph_graph",
    )
    monkeypatch.setattr(agent_tasks, "_owned_task", owned_task)
    monkeypatch.setattr(agent_tasks.repository, "get_task_run", get_task_run)
    monkeypatch.setattr(agent_tasks, "require_capability", require_capability)
    monkeypatch.setattr(agent_tasks, "get_runtime_registry", lambda: None)
    monkeypatch.setattr(agent_tasks.repository, "respond_to_budget_review", respond_to_budget_review)
    monkeypatch.setattr(agent_tasks.repository, "get_course_correction_command", get_command)
    monkeypatch.setattr(agent_tasks.repository, "mark_course_correction_delivered", mark_delivered)
    monkeypatch.setattr(agent_tasks.repository, "get_task", get_task)
    monkeypatch.setattr(agent_tasks, "definition_from_run", lambda _run: definition)
    monkeypatch.setattr(agent_tasks, "adapter_for_definition", lambda _definition: Adapter())

    response = await agent_tasks.respond_to_agent_task_budget_review(
        "task-1",
        agent_tasks.AgentTaskBudgetReviewRequest(
            run_id="run-1",
            interrupt_id="interrupt-1",
            expected_version=3,
            decision="steer",
            guidance="Investigate the remaining evidence.",
        ),
        thread_id="thread-1",
        idempotency_key="budget-steer-once",
    )

    assert len(submitted) == 1
    assert submitted[0][1].operation_id == "command-1"
    assert submitted[0][1].instruction == "Investigate the remaining evidence."
    assert delivered[0][0] == "command-1"
    assert response["task"]["current_phase"] == "budget_continuation_queued"
    assert response["correction_delivery"]["status"] == "accepted"
    assert response["linked_run"] is None
