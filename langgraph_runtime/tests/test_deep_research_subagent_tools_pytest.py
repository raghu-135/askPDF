from langgraph_runtime.workflows.deep_research_nodes import (
    _build_subagent_tool_input,
    _parse_subagent_action,
    _proposal_work_signature,
    _required_work_signature,
)
from langgraph_runtime.models.deep_research import DeepResearchPlanProposal, DeepResearchTodoProposal


def test_subagent_action_accepts_tool_name_as_action():
    decision = _parse_subagent_action(
        {"action": "inspect_document", "document_id": "file-hash-1"},
        ("inspect_document", "search_knowledge"),
    )
    assert decision.action == "tool"
    assert decision.tool == "inspect_document"
    tool_input = _build_subagent_tool_input(decision, "inspect_document", "fallback query")
    assert tool_input["document_id"] == "file-hash-1"
    assert "query" not in tool_input


def test_subagent_action_builds_query_for_search_tools():
    decision = _parse_subagent_action(
        {"action": "tool", "tool": "search_knowledge", "arguments": {"query": "transformer layers"}},
        ("search_knowledge",),
    )
    tool_input = _build_subagent_tool_input(decision, "search_knowledge", "fallback")
    assert tool_input["query"] == "transformer layers"


def test_required_work_signature_detects_identical_replan():
    prior = [{
        "profile_id": "document_researcher",
        "title": "Read paper A",
        "description": "Extract methods",
        "dependency_ids": [],
        "status": "failed",
    }]
    proposal = DeepResearchPlanProposal(
        objective="Compare papers",
        success_criteria=["Evidence"],
        todos=[
            DeepResearchTodoProposal(
                id="todo-1",
                title="Read paper A",
                description="Extract methods",
                completion_criteria="Done",
                dependency_ids=[],
                profile_id="document_researcher",
            ),
        ],
    )
    assert _proposal_work_signature(proposal) == _required_work_signature(prior)
