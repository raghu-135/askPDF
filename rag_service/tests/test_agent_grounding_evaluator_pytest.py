from types import SimpleNamespace

from app.services.agent_grounding_evaluator import AgentGroundingEvaluator


def test_ungrounded_document_when_available_result_is_incomplete():
    result = {
        "status": "completed",
        "runtime_task_result": {
            "status": "completed",
            "text": "An answer without retrieved evidence.",
            "warnings": [],
            "gaps": [],
        },
    }

    grounding = AgentGroundingEvaluator().evaluate(
        result,
        [SimpleNamespace(kind="operation.completed", payload_json={})],
        documents_present=False,
    )

    assert grounding["grounded"] is False
    assert grounding["evidence_result_count"] == 0
    assert grounding["successful_evidence_tools"] == []


def test_document_thread_requires_read_context_or_manifest_evidence():
    result = {"task_evidence_manifest": []}
    events = [
        SimpleNamespace(
            kind="tool.completed",
            payload_json={"ok": True, "tool_name": "search_knowledge", "result_count": 3},
        ),
    ]
    grounding = AgentGroundingEvaluator().evaluate(result, events, documents_present=True)
    assert grounding["grounded"] is False

    events_with_read = events + [
        SimpleNamespace(
            kind="tool.completed",
            payload_json={"ok": True, "tool_name": "read_context", "result_count": 1},
        ),
    ]
    grounding_ok = AgentGroundingEvaluator().evaluate(result, events_with_read, documents_present=True)
    assert grounding_ok["grounded"] is True
