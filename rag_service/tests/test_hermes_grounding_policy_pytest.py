from types import SimpleNamespace

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
