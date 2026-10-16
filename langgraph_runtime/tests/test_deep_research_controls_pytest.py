import json

from langgraph_runtime.adapter import _result_from_graph
from langgraph_runtime.models.deep_research import DeepResearchSubagentActionDecision
from langgraph_runtime.mcp_client import coerce_tool_arguments
from langgraph_runtime.workflows.deep_research_nodes import _ensure_web_research_todo
from langgraph_runtime.workflows.runtime_invocation import _usage_summary
from langgraph_runtime.models.deep_research import DeepResearchPlanProposal, DeepResearchTodoProposal, SubagentProfileId


def test_subagent_action_schema_has_no_additional_properties_on_arguments():
    schema = DeepResearchSubagentActionDecision.model_json_schema()
    properties = schema.get("properties") or {}
    assert "arguments" not in properties
    assert schema.get("additionalProperties") is False


def test_inspect_document_cursor_zero_is_coerced():
    payload = coerce_tool_arguments("inspect_document", {"document_id": "doc-1", "cursor": 0})
    assert isinstance(payload["cursor"], str)
    assert payload["cursor"]


def test_read_context_expansion_synonyms_are_coerced():
    payload = coerce_tool_arguments(
        "read_context",
        {"source_id": "src-1", "expansion": "full"},
    )
    assert payload["expansion"] == "section"
    payload = coerce_tool_arguments(
        "read_context",
        {"source_id": "src-1", "expansion": "expand"},
    )
    assert payload["expansion"] == "section"


def test_usage_summary_unwraps_structured_output_raw():
    class _Usage:
        prompt_tokens = 3
        completion_tokens = 7
        total_tokens = 10

    class _Raw:
        usage_metadata = _Usage()

    summary = _usage_summary({"raw": _Raw(), "parsed": {}})
    assert summary["total_tokens"] == 10


def test_fail_controller_route_projects_failed_runtime_status():
    result = _result_from_graph(
        {
            "agent_task_id": "task-1",
            "agent_run_id": "run-1",
            "task_version": 1,
            "task_controller_route": "fail",
            "task_controller_reason": "required_work_failed",
            "final_answer": "Deep research could not produce a usable plan.",
            "task_todos": [],
            "task_budget_usage": {},
        },
        operation_id="op-1",
    )
    assert result.status == "failed"
    assert result.orchestration_delta is not None
    assert result.orchestration_delta.result["status"] == "failed"


def test_ensure_web_research_todo_injects_when_mode_on():
    proposal = DeepResearchPlanProposal(
        objective="Test",
        success_criteria=["done"],
        todos=[
            DeepResearchTodoProposal(
                id="todo-1",
                title="Read docs",
                description="d",
                completion_criteria="c",
                dependency_ids=[],
                profile_id=SubagentProfileId.DOCUMENT,
            ),
        ],
    )
    updated = _ensure_web_research_todo(
        proposal,
        web_search_mode="ask",
        enabled_profiles=["document_researcher", "web_researcher"],
    )
    assert any(todo.profile_id == SubagentProfileId.WEB for todo in updated.todos)
