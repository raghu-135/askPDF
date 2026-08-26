from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.api.agent_tasks import _run_payload
from app.services import agent_task_runtime
from app.services.agent_task_runtime import _hermes_grounding_summary


def _event(kind, **payload):
    return SimpleNamespace(kind=kind, payload_json={"source": "askpdf_mcp", **payload})


def test_document_task_requires_nonempty_document_evidence():
    events = [
        _event("tool.failed", tool_name="search_document_by_id", error={"code": "tool_arguments_invalid"}),
        _event("tool.completed", tool_name="search_web", ok=True, result_count=4),
    ]
    summary = _hermes_grounding_summary(events, documents_present=True)
    assert summary["grounded"] is False
    assert summary["failure_codes"] == ["tool_arguments_invalid"]


def test_later_successful_document_retrieval_satisfies_grounding():
    events = [
        _event("tool.failed", tool_name="search_documents", error={"code": "tool_execution_failed"}),
        _event("tool.completed", tool_name="search_document_by_id", ok=True, result_count=3),
    ]
    summary = _hermes_grounding_summary(events, documents_present=True)
    assert summary["grounded"] is True
    assert summary["evidence_result_count"] == 3


def test_no_document_task_accepts_research_evidence_but_not_context_discovery():
    assert _hermes_grounding_summary([
        _event("tool.completed", tool_name="search_thread_conversation_history", ok=True, result_count=2),
    ], documents_present=False)["grounded"] is False
    assert _hermes_grounding_summary([
        _event("tool.completed", tool_name="wikipedia", ok=True, result_count=2),
    ], documents_present=False)["grounded"] is True


def test_agent_task_run_payload_preserves_required_trace_details():
    run = SimpleNamespace(
        id="run-1", task_id="task-1", task_attempt=1, parent_run_id=None,
        status="failed", checkpoint_thread_id=None, pending_interrupt_json=None,
        runtime_binding_status=None,
        metrics_json={}, error_json={"code": "runtime_limit_exceeded"},
        started_at=None, completed_at=None,
        debug_trace_json={"version": 1, "trace": {"status": "failed"}, "summary": {}, "details": [{"operation_id": "hermes"}]},
    )

    payload = _run_payload(run)

    assert payload["debug"]["trace"]["status"] == "failed"
    assert payload["debug"]["details"] == [{"operation_id": "hermes"}]


@pytest.mark.asyncio
async def test_terminal_run_and_trace_are_written_atomically(monkeypatch):
    repository = SimpleNamespace(complete_run=AsyncMock(return_value=SimpleNamespace(id="run-1")))
    run = SimpleNamespace(id="run-1", status="running", completed_at=None, debug_trace_json=None)
    monkeypatch.setattr(
        agent_task_runtime,
        "finalize_and_merge_debug_payload",
        lambda **kwargs: {"version": 1, "trace": {"status": kwargs["run_status"]}, "summary": {}},
    )

    await agent_task_runtime._complete_run_with_trace(
        repository,
        run=run,
        recorder=SimpleNamespace(),
        status="failed",
        metrics={"duration_ms": 300000},
        result={},
        error={"code": "runtime_limit_exceeded"},
    )

    repository.complete_run.assert_awaited_once()
    call = repository.complete_run.await_args
    assert call.kwargs["debug_trace_json"]["trace"]["status"] == "failed"
    assert call.kwargs["completed_at"] is not None
