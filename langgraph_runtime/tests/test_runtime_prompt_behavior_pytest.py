from langgraph_runtime.workflows.deep_research_nodes import DEEP_RESEARCH_POLICY, _deep_system
from langgraph_runtime.workflows.prompting import (
    build_agent_workflow_prompt_preview,
    build_planner_prompt,
    build_replanner_prompt,
    build_router_prompt,
)


def test_deep_research_runtime_policy_is_used_by_deep_nodes():
    assert DEEP_RESEARCH_POLICY
    assert DEEP_RESEARCH_POLICY in _deep_system("node-specific role")


def test_router_agent_prompt_preview_uses_runtime_prompts():
    prompt = build_agent_workflow_prompt_preview(
        workflow_id="router_rag_agent",
        context_window=8192,
        system_role="Expert AI Research Assistant specializing in analyzing uploaded documents and synthesizing accurate answers.",
        use_web_search=True,
        client_timezone="America/Chicago",
        client_locale="en-US",
    )
    assert "# Router Node Prompt" in prompt
    assert "# Final Answer Prompt" in prompt
    assert "{{QUESTION}}" in prompt


def test_plan_execute_agent_prompt_preview_uses_planner_prompt():
    prompt = build_agent_workflow_prompt_preview(workflow_id="plan_execute_rag_agent", context_window=8192)
    assert "# Planner Node Prompt" in prompt
    assert "# Final Answer Prompt" in prompt


def test_planner_and_replanner_prefer_comprehensive_relevant_worker_coverage():
    state = {
        "question": "Compare all relevant sources",
        "use_web_search": True,
        "pre_fetch_bundle": {},
        "available_worker_nodes": [
            {"id": "retrieval_worker", "type": "retrieval_worker"},
            {"id": "web_worker", "type": "web_worker"},
        ],
    }
    planner_prompt = build_planner_prompt(state)
    replanner_prompt = build_replanner_prompt(state)
    assert "Build a comprehensive retrieval plan" in planner_prompt
    assert "worker_decisions" in planner_prompt
    assert "Use as many relevant workers as needed" in replanner_prompt


def test_runtime_graph_prompt_builders_include_datetime_context():
    state = {"question": "What is the latest document?", "use_web_search": False, "client_timezone": "America/Chicago", "client_locale": "en-US", "pre_fetch_bundle": {}}
    assert "RUNTIME DATE/TIME CONTEXT" in build_router_prompt(state)
    assert "RUNTIME DATE/TIME CONTEXT" in build_planner_prompt(state)
