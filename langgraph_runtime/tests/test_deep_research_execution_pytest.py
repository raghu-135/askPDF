import asyncio
from typing import Annotated, TypedDict

import pytest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from langgraph_runtime.workflows.deep_research_execution import RuntimeExecutionServices, run_cancellable, runtime_execution_services_factory
from langgraph_runtime.workflows.state import (
    consume_task_result_packets,
    merge_task_result_packets,
    task_result_packet_identity,
)
from langgraph_runtime.workflows import deep_research_nodes
from runtime_protocol.errors import RuntimeError as AgentRuntimeError


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


def _packet(*, dispatch_id="dispatch-1", work_id="work-1", todo_id="todo-1", attempt=1, status="completed"):
    return {
        "dispatch_id": dispatch_id, "execution_key": work_id,
        "work_id": work_id, "attempt": attempt, "status": status,
        "todo_id": todo_id,
    }


def test_task_result_packet_consume_command_replaces_only_acknowledged_packets():
    first = _packet(work_id="work-1")
    retry = _packet(work_id="work-1", attempt=2)
    second = _packet(work_id="work-2")
    state = merge_task_result_packets([], [first, retry, second])

    consumed = consume_task_result_packets([task_result_packet_identity(first)])
    assert merge_task_result_packets(state, consumed) == [retry, second]
    assert merge_task_result_packets(state, consumed) == [retry, second]


def test_task_result_packet_identity_fails_closed_without_dispatch_work_identity():
    with pytest.raises(ValueError, match="dispatch/work identity"):
        task_result_packet_identity({"attempt": 1, "status": "completed"})


class _DeepWaveState(TypedDict, total=False):
    task_todos: list[dict]
    task_work_items: list[dict]
    task_result_packets: Annotated[list[dict], merge_task_result_packets]
    task_result_warnings: list[dict]
    task_result_gaps: list[str]
    task_web_access_decision: dict
    task_limits: dict
    task_run_plan_count: int
    task_controller_route: str
    task_controller_reason: str


class _NoopCancellation:
    async def requested(self):
        return False


class _CoordinatorServices:
    cancellation = _NoopCancellation()
    events = None

    def __init__(self, state):
        self.state = state
        self.recorded: list[dict] = []

    async def record_result_packets(self, packets):
        self.recorded.extend(packets)
        todos = [dict(todo) for todo in self.state.get("task_todos") or []]
        by_id = {str(todo["id"]): todo for todo in todos}
        for packet in packets:
            todo = by_id[str(packet["todo_id"])]
            if packet["status"] == "completed":
                todo["status"] = "completed"
            elif packet.get("retryable"):
                todo["status"] = "ready"
        self.state["task_todos"] = todos
        return todos

    async def assemble_artifact_context(self, _compact):
        return {}

    async def pause_requested(self):
        return False

    async def budget_boundary(self):
        return None

    async def budget_snapshot(self):
        return {}

    async def pending_course_corrections(self):
        return []

    async def persist_web_access(self, *_args, **_kwargs):
        return None


@pytest.mark.asyncio
async def test_compiled_deep_coordinator_consumes_packets_between_dispatch_waves(monkeypatch):
    services_by_state = {}

    def services_factory(_config, state):
        key = id(state)
        return services_by_state.setdefault(key, _CoordinatorServices(state))

    monkeypatch.setattr(deep_research_nodes, "services_from_config", services_factory)

    async def worker_a(_state):
        return {"task_result_packets": [_packet(work_id="work-a", todo_id="todo-a")], "task_work_items": []}

    async def worker_b(_state):
        return {"task_result_packets": [_packet(work_id="work-b", todo_id="todo-b")], "task_work_items": []}

    graph = StateGraph(_DeepWaveState)
    graph.add_node("worker_a", worker_a)
    graph.add_node("worker_b", worker_b)
    graph.add_node("coordinator", deep_research_nodes.deep_coordinator)
    graph.add_edge(START, "worker_a")
    graph.add_edge("worker_a", "coordinator")
    graph.add_conditional_edges(
        "coordinator",
        lambda state: "worker_b" if state["task_controller_route"] == "dispatch_more" else END,
        {"worker_b": "worker_b", END: END},
    )
    graph.add_edge("worker_b", "coordinator")
    app = graph.compile()
    initial = {
        "task_todos": [
            {"id": "todo-a", "status": "running", "attempt": 1, "dependency_ids": []},
            {"id": "todo-b", "status": "pending", "attempt": 1, "dependency_ids": ["todo-a"]},
        ],
        "task_work_items": [{"dispatch_id": "dispatch-1"}],
        "task_result_packets": [], "task_result_warnings": [], "task_result_gaps": [],
    }

    result = await app.ainvoke(initial)

    assert [packet["work_id"] for packet in sum((service.recorded for service in services_by_state.values()), [])] == ["work-a", "work-b"]
    assert result["task_result_packets"] == []
    assert {todo["status"] for todo in result["task_todos"]} == {"completed"}


@pytest.mark.asyncio
async def test_compiled_deep_coordinator_keeps_retry_attempts_distinct(monkeypatch):
    services_by_state = {}

    def services_factory(_config, state):
        return services_by_state.setdefault(id(state), _CoordinatorServices(state))

    monkeypatch.setattr(deep_research_nodes, "services_from_config", services_factory)

    async def first_worker(_state):
        return {"task_result_packets": [{**_packet(work_id="work-retry", todo_id="todo-retry", attempt=1, status="failed"), "retryable": True}], "task_work_items": []}

    async def prepare_retry(_state):
        return {
            "task_todos": [{"id": "todo-retry", "status": "running", "attempt": 2, "dependency_ids": []}],
            "task_work_items": [{"dispatch_id": "dispatch-2"}],
        }

    async def second_worker(_state):
        return {"task_result_packets": [_packet(dispatch_id="dispatch-2", work_id="work-retry", todo_id="todo-retry", attempt=2)], "task_work_items": []}

    graph = StateGraph(_DeepWaveState)
    graph.add_node("first_worker", first_worker)
    graph.add_node("prepare_retry", prepare_retry)
    graph.add_node("second_worker", second_worker)
    graph.add_node("coordinator", deep_research_nodes.deep_coordinator)
    graph.add_edge(START, "first_worker")
    graph.add_edge("first_worker", "coordinator")
    graph.add_conditional_edges(
        "coordinator",
        lambda state: "prepare_retry" if state["task_controller_route"] == "dispatch_more" else END,
        {"prepare_retry": "prepare_retry", END: END},
    )
    graph.add_edge("prepare_retry", "second_worker")
    graph.add_edge("second_worker", "coordinator")

    result = await graph.compile().ainvoke({
        "task_todos": [{"id": "todo-retry", "status": "running", "attempt": 1, "dependency_ids": []}],
        "task_work_items": [{"dispatch_id": "dispatch-1"}],
        "task_result_packets": [], "task_result_warnings": [], "task_result_gaps": [],
    })

    recorded = sum((service.recorded for service in services_by_state.values()), [])
    assert [(packet["attempt"], packet["dispatch_id"]) for packet in recorded] == [(1, "dispatch-1"), (2, "dispatch-2")]
    assert result["task_result_packets"] == []
    assert result["task_todos"][0]["status"] == "completed"


@pytest.mark.asyncio
async def test_compiled_packet_consumption_survives_checkpoint_continuation():
    class State(TypedDict, total=False):
        task_result_packets: Annotated[list[dict], merge_task_result_packets]

    def produce(_state):
        return {"task_result_packets": [_packet()]}

    def consume(state):
        interrupt("before-consume")
        return {"task_result_packets": consume_task_result_packets([
            task_result_packet_identity(state["task_result_packets"][0])
        ])}

    graph = StateGraph(State)
    graph.add_node("produce", produce)
    graph.add_node("consume", consume)
    graph.add_edge(START, "produce")
    graph.add_edge("produce", "consume")
    graph.add_edge("consume", END)
    app = graph.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "packet-continuation"}}

    paused = await app.ainvoke({"task_result_packets": []}, config=config)
    assert paused["__interrupt__"]
    resumed = await app.ainvoke(Command(resume=True), config=config)
    assert resumed["task_result_packets"] == []


@pytest.mark.asyncio
async def test_failed_provisional_synthesis_keeps_answer_empty_and_disables_acceptance(monkeypatch):
    captured = {}

    def fake_interrupt(payload):
        captured.update(payload)
        return {"action": "continue"}

    monkeypatch.setattr(deep_research_nodes, "interrupt", fake_interrupt)
    state = {
        "agent_task_id": "task-1", "agent_run_id": "run-1", "final_answer": "",
        "task_provisional_synthesis_failed": {"code": "provisional_synthesis_failed"},
        "task_budget_boundary": {"status": "requested", "dimensions": ["model_calls"]},
        "task_incomplete_reasons": ["synthesis_unavailable"], "task_evidence_manifest": [],
        "task_limits": {"max_model_calls": 10, "max_model_tokens": 1000, "max_tool_calls": 10, "max_active_runtime_ms": 1000},
        "task_budget_usage": {"tranche_index": 1, "tranche_limits": {"model_calls": 10, "model_tokens": 1000, "tool_calls": 10, "elapsed_active_ms": 1000}, "tranche_usage": {"model_calls": 0, "model_tokens": 0, "tool_calls": 0, "elapsed_active_ms": 0}, "lifetime_usage": {"model_calls": 0, "model_tokens": 0, "tool_calls": 0, "elapsed_active_ms": 0}},
    }

    result = await deep_research_nodes.evidence_critic(state, {"configurable": {
        "deep_research_services_factory": runtime_execution_services_factory,
        "cancellation_checker": lambda: False,
    }})

    assert result["final_answer"] == ""
    assert captured["allowed_actions"] == ["continue", "steer"]
    assert captured["provisional_answer"] == ""
    assert captured["accept_partial_enabled"] is False
    assert result["task_budget_review_route"] == "continue"


@pytest.mark.asyncio
async def test_failed_provisional_synthesis_rejects_internal_partial_acceptance(monkeypatch):
    monkeypatch.setattr(deep_research_nodes, "interrupt", lambda _payload: {"action": "accept_partial"})
    state = {
        "agent_task_id": "task-1", "agent_run_id": "run-1", "final_answer": "",
        "task_provisional_synthesis_failed": {"code": "provisional_synthesis_failed"},
        "task_budget_boundary": {"status": "requested", "dimensions": ["model_calls"]},
        "task_incomplete_reasons": [], "task_evidence_manifest": [],
        "task_limits": {"max_model_calls": 10, "max_model_tokens": 1000, "max_tool_calls": 10, "max_active_runtime_ms": 1000},
        "task_budget_usage": {"tranche_index": 1, "tranche_limits": {"model_calls": 10, "model_tokens": 1000, "tool_calls": 10, "elapsed_active_ms": 1000}, "tranche_usage": {"model_calls": 0, "model_tokens": 0, "tool_calls": 0, "elapsed_active_ms": 0}, "lifetime_usage": {"model_calls": 0, "model_tokens": 0, "tool_calls": 0, "elapsed_active_ms": 0}},
    }

    with pytest.raises(AgentRuntimeError, match="No usable provisional answer"):
        await deep_research_nodes.evidence_critic(state, {"configurable": {
            "deep_research_services_factory": runtime_execution_services_factory,
            "cancellation_checker": lambda: False,
        }})
