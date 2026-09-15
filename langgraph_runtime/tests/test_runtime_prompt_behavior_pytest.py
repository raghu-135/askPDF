import pytest

from langgraph_runtime.prompts.loaders import (
    RUNTIME_PROMPT_FILES,
    get_deep_research_policy,
    get_web_search_mandate,
    runtime_prompt_path,
    validate_runtime_prompt_assets,
)
from langgraph_runtime.workflows.deep_research_nodes import DEEP_RESEARCH_POLICY, _deep_system
from langgraph_runtime.workflows.prompting import (
    build_agent_workflow_prompt_preview,
    build_evaluator_prompt,
    build_final_answer_messages,
    build_grounded_answer_verifier_prompt,
    build_planner_prompt,
    build_replanner_prompt,
    build_retrieval_quality_prompt,
    build_router_prompt,
)


def test_deep_research_runtime_policy_is_used_by_deep_nodes():
    assert DEEP_RESEARCH_POLICY
    assert DEEP_RESEARCH_POLICY in _deep_system("node-specific role")
    assert "Canvas publish is runtime-owned at synthesis" in DEEP_RESEARCH_POLICY


def test_runtime_prompt_manifest_is_complete_and_runtime_owned():
    validate_runtime_prompt_assets()
    assert RUNTIME_PROMPT_FILES
    assert all(runtime_prompt_path(filename).is_file() for filename in RUNTIME_PROMPT_FILES)
    assert all("product_orchestration" not in str(runtime_prompt_path(filename)) for filename in RUNTIME_PROMPT_FILES)
    assert get_deep_research_policy()
    assert get_web_search_mandate()


def test_runtime_prompt_validation_reports_missing_assets(monkeypatch, tmp_path):
    import langgraph_runtime.prompts.loaders as loaders

    monkeypatch.setattr(loaders, "PROMPTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="deep_research_policy.md"):
        loaders.validate_runtime_prompt_assets()


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


def test_canvas_layout_skills_gate_on_publish_admission():
    off = build_router_prompt({"question": "Compare the papers", "use_web_search": False, "pre_fetch_bundle": {}})
    assert "canvas_compare_papers" not in off
    assert "canvas_evidence_matrix" not in off
    assert "canvas_timeline" not in off
    assert "publish_canvas" not in off

    on_router = build_router_prompt(
        {
            "question": "Compare the papers",
            "use_web_search": False,
            "pre_fetch_bundle": {},
            "allowed_tool_ids": ["research_canvas_publish"],
        }
    )
    assert "canvas_compare_papers" not in on_router
    assert "publish_canvas" not in on_router

    on_answer = build_final_answer_messages(
        {
            "question": "Compare the papers",
            "use_web_search": False,
            "pre_fetch_bundle": {},
            "allowed_tool_ids": ["research_canvas_publish"],
        },
        "Evidence context",
    )
    combined = on_answer["system"] + "\n" + on_answer["human"]
    assert "canvas_compare_papers" in combined
    assert "canvas_evidence_matrix" in combined
    assert "canvas_timeline" in combined
    assert "publish_canvas" in combined


def test_runtime_graph_prompt_builders_include_datetime_context():
    state = {"question": "What is the latest document?", "use_web_search": False, "client_timezone": "America/Chicago", "client_locale": "en-US", "pre_fetch_bundle": {}}
    assert "RUNTIME DATE/TIME CONTEXT" in build_router_prompt(state)
    assert "RUNTIME DATE/TIME CONTEXT" in build_planner_prompt(state)


def test_all_runtime_prompt_builders_resolve_runtime_owned_assets():
    state = {
        "question": "Explain the evidence",
        "use_web_search": True,
        "pre_fetch_bundle": {"document_evidence_text": "A useful excerpt"},
        "evaluator_report": "Needs more support",
    }
    assert build_evaluator_prompt(state)
    assert build_retrieval_quality_prompt(state)
    assert build_grounded_answer_verifier_prompt(state)
    final_messages = build_final_answer_messages(state, "Evidence context")
    assert final_messages["system"]
    assert final_messages["human"]
