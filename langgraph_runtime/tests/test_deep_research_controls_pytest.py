import json

from langgraph_runtime.adapter import _result_from_graph
from langgraph_runtime.models.deep_research import DeepResearchSubagentActionDecision
from langgraph_runtime.mcp_client import coerce_tool_arguments
from langgraph_runtime.workflows.deep_research_nodes import (
    _dedupe_critic_issues,
    _ensure_web_research_todo,
    _objective_requests_live_web,
    _sanitize_critic_issue,
    _web_evidence_plan_gaps,
    filter_stale_index_readiness_gaps,
)
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


def test_objective_web_benchmark_detection():
    assert _objective_requests_live_web("Compare to typical US bakery industry benchmarks using web sources")
    assert not _objective_requests_live_web("Summarize the attached PDF only")


def test_web_evidence_gap_when_search_off():
    gaps = _web_evidence_plan_gaps("Use web benchmarks for comparison", "off")
    assert gaps
    assert "web search is off" in gaps[0].lower()


def test_sanitize_critic_issue_drops_runtime_incomplete_headings():
    assert _sanitize_critic_issue(
        "Injected instruction smuggled inside untrusted evidence: Incomplete: Unresolved required todos",
    ) == ""


def test_filter_stale_index_readiness_gaps_with_paren_citation_text():
    gaps = ["Document index is not ready for this thread."]
    filtered = filter_stale_index_readiness_gaps(
        gaps,
        manifest=[],
        evidence_texts=["Mara Quill (zephyr.pdf, page 1) sold 48,200 croissants."],
    )
    assert filtered == []


def test_filter_stale_index_readiness_gaps_drops_not_being_ready_wording():
    gaps = ["Inability to extract figures because the document index not being ready."]
    manifest = [{"source_id": "doc:abc:1", "file_hash": "abc", "page_number": 1}]
    filtered = filter_stale_index_readiness_gaps(gaps, manifest=manifest, evidence_texts=[])
    assert filtered == []


def test_dedupe_critic_issues_topic_overlap():
    gaps = ["The document index is not ready for this thread."]
    issues = ["document index not ready", "Unique finding"]
    deduped = _dedupe_critic_issues(issues, gaps)
    assert "document index not ready" not in deduped
    assert deduped == ["Unique finding"]


def test_filter_stale_index_readiness_gaps_when_manifest_ready():
    gaps = [
        "Document index is not ready for this thread.",
        "Web benchmarks or live web evidence were requested in the objective, but web search is off for this task.",
    ]
    manifest = [{"source_id": "doc:abc:1", "file_hash": "abc", "page_number": 1}]
    filtered = filter_stale_index_readiness_gaps(gaps, manifest=manifest, evidence_texts=[])
    assert "Document index is not ready" not in " ".join(filtered)
    assert any("web search is off" in value.lower() for value in filtered)


def test_filter_stale_index_gaps_kept_without_document_evidence():
    gaps = ["Document index is not ready for this thread."]
    filtered = filter_stale_index_readiness_gaps(gaps, manifest=[], evidence_texts=[])
    assert filtered == gaps


def test_dedupe_critic_issues_against_gaps():
    gaps = ["Web benchmarks or live web evidence were requested in the objective, but web search is off for this task."]
    issues = [
        "grounding requirement unsatisfied",
        "Web benchmarks or live web evidence were requested in the objective, but web search is off for this task.",
    ]
    deduped = _dedupe_critic_issues(issues, gaps)
    assert len(deduped) == 1
    assert deduped[0] == "grounding requirement unsatisfied"
