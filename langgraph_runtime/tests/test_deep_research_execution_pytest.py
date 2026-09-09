import asyncio

import pytest

from langgraph_runtime.workflows.deep_research_execution import RuntimeExecutionServices, run_cancellable


class Token:
    def __init__(self):
        self.cancelled = False

    async def requested(self):
        return self.cancelled


@pytest.mark.asyncio
async def test_run_cancellable_stops_blocking_work():
    token = Token()
    stopped = asyncio.Event()

    async def work():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    operation = asyncio.create_task(run_cancellable(work(), token, poll_seconds=0.001))
    await asyncio.sleep(0)
    token.cancelled = True
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert stopped.is_set()


@pytest.mark.asyncio
async def test_run_cancellable_cleans_up_on_timeout():
    stopped = asyncio.Event()

    async def work():
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    with pytest.raises(asyncio.TimeoutError):
        await run_cancellable(work(), Token(), timeout_seconds=0.001, poll_seconds=0.001)
    assert stopped.is_set()


def _services(todos):
    return RuntimeExecutionServices(
        todos=None,
        artifacts=None,
        budgets=None,
        cancellation=Token(),
        events=None,
        memory=None,
        state={"task_todos": todos},
    )


@pytest.mark.asyncio
async def test_scheduler_persists_claimed_attempt_and_stops_at_max_attempts():
    services = _services([{
        "id": "todo-1", "status": "pending", "priority": 1,
        "dependency_ids": [], "attempt": 0, "max_attempts": 2,
    }])

    first = await services.schedule_ready("task-1", limit=1)
    assert first[0].status == "running"
    assert first[0].attempt == 1
    assert services.state["task_todos"][0]["status"] == "running"
    assert services.state["task_todos"][0]["attempt"] == 1

    await services.record_result_packets([{
        "todo_id": "todo-1", "attempt": 1, "status": "failed",
        "retryable": True, "summary": "first failure",
    }])
    assert services.state["task_todos"][0]["status"] == "ready"
    assert services.state["task_todos"][0]["attempt"] == 1

    second = await services.schedule_ready("task-1", limit=1)
    assert second[0].attempt == 2
    assert services.state["task_todos"][0]["attempt"] == 2

    await services.record_result_packets([{
        "todo_id": "todo-1", "attempt": 2, "status": "failed",
        "retryable": True, "summary": "final failure",
    }])
    assert services.state["task_todos"][0]["status"] == "failed"
    assert services.state["task_todos"][0]["result_summary"] == "max_attempts_exhausted"
    assert await services.schedule_ready("task-1", limit=1) == []


@pytest.mark.asyncio
async def test_scheduler_enforces_dependencies_and_deterministic_priority():
    services = _services([
        {"id": "dependent", "status": "pending", "priority": 100, "dependency_ids": ["first"]},
        {"id": "independent", "status": "pending", "priority": 10, "dependency_ids": []},
        {"id": "first", "status": "pending", "priority": 1, "dependency_ids": []},
    ])

    first_batch = await services.schedule_ready("task-1", limit=2)
    assert [todo.id for todo in first_batch] == ["independent", "first"]
    assert services.state["task_todos"][0]["status"] == "pending"

    await services.record_result_packets([{
        "todo_id": "first", "attempt": 1, "status": "completed",
        "retryable": False, "summary": "prerequisite complete",
    }])
    second_batch = await services.schedule_ready("task-1", limit=1)
    assert [todo.id for todo in second_batch] == ["dependent"]


@pytest.mark.asyncio
async def test_scheduler_blocks_dependents_of_failed_prerequisites():
    services = _services([
        {"id": "first", "status": "failed", "dependency_ids": []},
        {"id": "second", "status": "pending", "dependency_ids": ["first"]},
        {"id": "third", "status": "pending", "dependency_ids": ["second"]},
    ])

    assert await services.schedule_ready("task-1", limit=2) == []
    assert services.state["task_todos"][1]["status"] == "blocked"
    assert services.state["task_todos"][2]["status"] == "blocked"
