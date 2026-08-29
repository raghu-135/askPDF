from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services import agent_run_cancellation as cancellation
from app.runtime.errors import RuntimeError as AgentRuntimeError


class Adapter:
    def __init__(self):
        self.requests = []

    async def cancel(self, request):
        self.requests.append(request)
        return {"status": "cancellation_requested"}


class TerminalAdapter(Adapter):
    async def cancel(self, request):
        self.requests.append(request)
        return {"status": "completed", "no_op": True}


class Registry:
    def __init__(self, adapter):
        self.adapter = adapter

    def get(self, _definition):
        return self.adapter


def _run():
    return SimpleNamespace(
        id="run-1", thread_id="thread-1", workflow_id="deep_research_agent",
        framework="langgraph", builder_id="langgraph_graph", definition_category="task",
        runtime_binding_json={}, runtime_binding_status="active", status="running", task_id="task-1",
    )


@pytest.mark.asyncio
async def test_active_task_cancellation_is_submitted_but_not_terminal(monkeypatch):
    capability = AsyncMock()
    monkeypatch.setattr(cancellation, "require_capability", capability)
    adapter = Adapter()
    registry = Registry(adapter)
    task = SimpleNamespace(id="task-1", status="cancelling")
    run = _run()
    result = await cancellation.request_task_cancellation(task, run, registry=registry)
    assert result["status"] == "cancelling"
    assert result["runtime_confirmation"] == "pending"
    assert len(adapter.requests) == 1
    capability.assert_awaited_once()
    assert capability.await_args.kwargs == {"registry": registry, "run": run}


@pytest.mark.asyncio
async def test_terminal_runtime_cancellation_response_preserves_terminal_status(monkeypatch):
    monkeypatch.setattr(cancellation, "require_capability", AsyncMock())
    result = await cancellation.request_task_cancellation(
        SimpleNamespace(id="task-1", status="cancelling"),
        _run(),
        registry=Registry(TerminalAdapter()),
    )

    assert result["status"] == "cancelling"
    assert result["runtime_status"] == "completed"
    assert result["runtime_confirmation"] == "terminal"


@pytest.mark.asyncio
async def test_task_without_runtime_cancels_without_adapter():
    result = await cancellation.request_task_cancellation(SimpleNamespace(id="task-1"), None)
    assert result == {"status": "cancelled", "task_id": "task-1", "runtime_confirmation": "not_required"}


@pytest.mark.asyncio
async def test_unsupported_cancellation_fails_before_adapter_invocation(monkeypatch):
    adapter = Adapter()
    unsupported = AgentRuntimeError.capability_unsupported(
        operation_id="run.cancel", framework="langgraph", builder_id="langgraph_graph",
        explanation="disabled for this definition",
    )
    monkeypatch.setattr(cancellation, "require_capability", AsyncMock(side_effect=unsupported))
    task = SimpleNamespace(id="task-1", status="running")
    run = _run()

    with pytest.raises(AgentRuntimeError) as caught:
        await cancellation.request_task_cancellation(task, run, registry=Registry(adapter))

    assert caught.value.code == "runtime_capability_unsupported"
    assert adapter.requests == []


@pytest.mark.asyncio
async def test_transport_failure_is_retryable_and_nonterminal(monkeypatch):
    monkeypatch.setattr(cancellation, "require_capability", AsyncMock())
    adapter = Adapter()
    adapter.cancel = AsyncMock(side_effect=OSError("connection reset"))

    with pytest.raises(AgentRuntimeError) as caught:
        await cancellation.request_task_cancellation(
            SimpleNamespace(id="task-1", status="cancelling"),
            _run(),
            registry=Registry(adapter),
        )

    assert caught.value.code == "runtime_transport_error"
    assert caught.value.retryable is True


@pytest.mark.asyncio
async def test_confirmed_cancellation_atomically_terminalizes_task_and_run(monkeypatch):
    record = AsyncMock()
    finalize = AsyncMock(return_value=SimpleNamespace(status="cancelled"))
    complete_commands = AsyncMock()
    monkeypatch.setattr("app.services.agent_runtime_reconciliation.record_terminal_result", record)
    monkeypatch.setattr("app.services.agent_task_repository.finalize_task_run", finalize)
    monkeypatch.setattr("app.services.agent_task_repository.complete_pending_cancel_commands", complete_commands)
    task = SimpleNamespace(id="task-1")
    run = _run()
    run.metrics_json = {"model_calls": 1}
    run.debug_trace_json = {"events": []}

    result = await cancellation.confirm_task_cancellation(
        task, run, result={"status": "cancelled"}, terminal_event_id="terminal-1"
    )

    assert result.status == "cancelled"
    record.assert_awaited_once()
    kwargs = finalize.await_args.kwargs
    assert kwargs["run_status"] == kwargs["task_status"] == "cancelled"
    assert kwargs["terminal_event"].kind == "run.cancelled"
    assert kwargs["terminal_event"].terminal is True
    complete_commands.assert_awaited_once()
    assert complete_commands.await_args.kwargs["result"]["runtime_confirmation"] == "confirmed"
