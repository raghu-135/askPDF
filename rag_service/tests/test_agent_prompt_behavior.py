"""Regression tests for graph runtime prompt behavior."""

from datetime import datetime, timezone


from app.agent.prompting import format_runtime_datetime_context
from app.agent_patterns.prompting import build_agent_pattern_prompt_preview, build_planner_prompt, build_router_prompt


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


def test_router_agent_prompt_preview_uses_graph_runtime_prompts():
    prompt = build_agent_pattern_prompt_preview(
        pattern_id="router_rag_agent",
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
    assert "Timeline retrieval should preserve temporal anchor words" in prompt
    assert "Assistant role:" in prompt
    assert "Expert AI Research Assistant specializing in analyzing uploaded documents" in prompt
    assert "Planner Node Prompt" not in prompt


def test_plan_execute_agent_prompt_preview_uses_planner_prompt():
    prompt = build_agent_pattern_prompt_preview(
        pattern_id="plan_execute_rag_agent",
        context_window=8192,
    )

    assert "# Planner Node Prompt" in prompt
    assert "# Final Answer Prompt" in prompt
    assert "execution_plan" in prompt
    assert "include `timeline_worker`" in prompt
    assert "For prior conversation recall without time/order wording" in prompt
    assert "Clarification options must contain 2-4 complete, self-contained questions" in prompt
    assert "Choose `direct` only when pre-fetched context directly answers the question" in prompt
    assert "`retrieval_worker` queries should preserve named files, pages, sections, citations, or quoted text" in prompt
    assert "`web_worker` queries should use concise keyword-rich queries" in prompt


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
