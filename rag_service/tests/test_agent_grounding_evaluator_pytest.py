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
