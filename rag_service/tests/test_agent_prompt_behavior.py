"""Regression tests for graph runtime prompt behavior."""

from datetime import datetime, timezone


from app.agent.prompting import get_tool_catalog, normalize_tool_instructions, format_runtime_datetime_context
from langgraph_runtime.workflows.prompting import (
    build_agent_workflow_prompt_preview,
    build_planner_prompt,
    build_replanner_prompt,
    build_router_prompt,
)
from langgraph_runtime.workflows.deep_research_nodes import DEEP_RESEARCH_POLICY, _deep_system
from app.prompts.loaders import get_deep_research_policy


def test_runtime_datetime_context_uses_browser_timezone_with_server_clock():
    """Runtime clock should be computed in the browser timezone from server UTC."""
    context = format_runtime_datetime_context(
        client_timezone="America/Chicago",
        client_locale="en-US",
        client_now_iso="2026-06-25T19:00:00.000Z",
        now_utc=datetime(2026, 6, 25, 19, 0, tzinfo=timezone.utc),
    )

    assert "RUNTIME DATE/TIME CONTEXT" in context
    assert "User timezone: America/Chicago" in context
    assert "User locale: en-US" in context
    assert "User-local current datetime: 2026-06-25T14:00:00-05:00" in context
    assert "Server current UTC datetime: 2026-06-25T19:00:00Z" in context


def test_deep_research_nodes_share_the_versioned_policy_with_hermes():
    policy = get_deep_research_policy("deep_research_v1")
    assert DEEP_RESEARCH_POLICY == policy
    assert policy in _deep_system("node-specific role")
    assert "successful, nonempty document-retrieval result" in policy


def test_router_agent_prompt_preview_uses_graph_runtime_prompts():
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
    assert "Temporal Metadata Contract" in prompt
    assert "Tool Playbook" in prompt
    assert "Web Search Mandate" in prompt
    assert "{{QUESTION}}" in prompt
    assert "Choose `direct` only when pre-fetched context directly answers the question" in prompt
    assert "Do not choose `direct` for latest, first, since, before, after, or current questions" in prompt
    assert "Document retrieval should preserve named files, pages, sections, citations, or quoted text" in prompt
    assert "`thread_events_worker` should preserve temporal anchor words" in prompt
    assert "Assistant role:" in prompt
    assert "Expert AI Research Assistant specializing in analyzing uploaded documents" in prompt
    assert "Planner Node Prompt" not in prompt


def test_plan_execute_agent_prompt_preview_uses_planner_prompt():
    prompt = build_agent_workflow_prompt_preview(
        workflow_id="plan_execute_rag_agent",
        context_window=8192,
    )

    assert "# Planner Node Prompt" in prompt
    assert "# Final Answer Prompt" in prompt
    assert "worker_decisions" in prompt
    assert "exactly one object for every available worker" in prompt
    assert "Do not select workers by keyword matching alone" in prompt
    assert "Clarification options must contain 2-4 complete, self-contained questions" in prompt
    assert "Choose `direct` only when pre-fetched context directly answers the question" in prompt
    assert "Include every available worker that has a reasonable chance" in prompt
    assert "do not minimize the worker count merely to reduce retrieval calls" in prompt


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
    assert "When uncertain whether a relevant worker could help, include it" in planner_prompt
    assert "worker_decisions" in planner_prompt
    assert "exactly one object for every available worker" in planner_prompt
    assert "Use as many relevant workers as needed to address the gaps comprehensively" in replanner_prompt
    assert "worker_decisions" in replanner_prompt
    assert "Prefer the smallest plan" not in replanner_prompt


def test_runtime_graph_prompt_builders_include_datetime_context():
    state = {
        "question": "What is the latest document?",
        "use_web_search": False,
        "client_timezone": "America/Chicago",
        "client_locale": "en-US",
        "pre_fetch_bundle": {},
    }

    assert "RUNTIME DATE/TIME CONTEXT" in build_router_prompt(state)
    assert "RUNTIME DATE/TIME CONTEXT" in build_planner_prompt(state)


def test_tool_catalog_and_legacy_instruction_keys_use_canonical_retrieval_names():
    catalog = {item["tool_name"]: item for item in get_tool_catalog()}

    assert catalog["search_thread_conversation_history"]["id"] == "thread_conversation_history"
    assert catalog["search_durable_memory"]["id"] == "durable_memory"
    assert catalog["search_thread_events"]["id"] == "thread_events"

    normalized = normalize_tool_instructions(
        {
            "deep_memory": "legacy instruction",
            "thread_conversation_history": "canonical instruction",
            "memory_recall": "durable instruction",
            "thread_timeline": "events instruction",
        }
    )

    assert normalized["thread_conversation_history"] == "canonical instruction"
    assert normalized["durable_memory"] == "durable instruction"
    assert normalized["thread_events"] == "events instruction"
