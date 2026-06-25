import pytest
from langchain_core.tools import tool

from app.agent import external_research_tools
from app.agent.agent import OrchestratorToolNode, format_intent_tool_context
from app.prompts.loaders import get_web_search_mandate, load_prompt
from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG


TOOL_PACKAGE_PINS = {
    "langgraph": "1.2.6",
    "langchain-core": "1.4.8",
    "langchain-community": "0.4.2",
    "ddgs": "9.14.4",
    "wikipedia": "1.4.0",
    "mediawikiapi": "1.3",
    "wikibase-rest-api-client": "0.2.5",
    "arxiv": "2.4.1",
    "xmltodict": "1.0.4",
    "yfinance": "1.4.1",
    "stackapi": "0.3.1",
    "semanticscholar": "0.12.0",
}


def _requirements_lines() -> set[str]:
    requirements_path = external_research_tools.__file__.split("/app/agent/")[0]
    with open(f"{requirements_path}/requirements.txt", encoding="utf-8") as req_file:
        return {
            line.strip()
            for line in req_file
            if line.strip() and not line.lstrip().startswith("#")
        }


def test_tool_dependencies_are_exactly_pinned():
    requirements = _requirements_lines()
    missing = {
        f"{package}=={version}"
        for package, version in TOOL_PACKAGE_PINS.items()
        if f"{package}=={version}" not in requirements
    }

    assert not missing


def test_external_research_tool_candidates_exclude_searxng(monkeypatch):
    """SearXNG-backed tools should not be registered in this lightweight expansion."""
    seen = []

    def fake_build_tool(display_name, tool_path, class_name, factory=None):
        seen.append((display_name, tool_path, class_name))
        return None

    monkeypatch.setattr(external_research_tools, "_build_tool", fake_build_tool)

    assert external_research_tools.get_external_research_tools() == []

    display_names = {item[0] for item in seen}
    assert display_names == {
        "Wikipedia",
        "Wikidata",
        "arXiv",
        "PubMed",
        "Semantic Scholar",
        "StackExchange",
        "Yahoo Finance News",
    }
    assert all("searx" not in tool_path.lower() for _, tool_path, _ in seen)


def test_external_research_tools_have_prompt_metadata():
    expected_tool_names = {
        "wikipedia",
        "wikidata",
        "arxiv",
        "pub_med",
        "pubmed",
        "semanticscholar",
        "semantic_scholar",
        "stack_exchange",
        "yahoo_finance_news",
        "search_web_intent",
        "search_thread_timeline",
    }

    missing = expected_tool_names - set(TOOL_FRIENDLY_CONFIG)
    assert not missing
    assert "find_topic_anchor_in_history" not in TOOL_FRIENDLY_CONFIG


def test_yahoo_finance_news_guidance_requires_ticker_and_search_web_prereq():
    prompt = TOOL_FRIENDLY_CONFIG["yahoo_finance_news"]["default_prompt"].lower()

    assert "only the ticker symbol" in prompt
    assert "do not pass a company name" in prompt
    assert "first call search_web" in prompt
    assert "if no public ticker exists, do not call this tool" in prompt


def test_search_web_guidance_stays_general_purpose():
    prompt = TOOL_FRIENDLY_CONFIG["search_web"]["default_prompt"].lower()

    assert "outside the uploaded documents" in prompt
    assert "likely time-sensitive" in prompt
    assert "yahoo" not in prompt
    assert "ticker" not in prompt
    assert "prerequisite" not in prompt


def test_web_search_mandate_allows_source_specific_tools():
    mandate = get_web_search_mandate().lower()

    assert "call search_web for every factual" in mandate
    assert "source-specific external tool" not in mandate
    assert "instead of substituting search_web" not in mandate


def test_intent_prompt_preserves_explicit_source_requests():
    prompt = load_prompt("intent_agent/system.md").lower()

    assert "orchestrator has its own tool catalog" in prompt
    assert "{intent_tool_context}" in prompt
    role_section = prompt.split("## output contract", 1)[0]
    assert "{intent_tool_context}" not in role_section
    assert "{orchestrator_tool_context}" not in role_section
    assert "there are two separate tool scopes" in prompt
    assert "intent agent tool catalog (callable by you now)" in prompt
    assert "downstream orchestrator tool catalog (not callable by you)" in prompt
    assert "preserve explicit source, connector, or tool constraints" in prompt
    assert "handoff constraints for the orchestrator's tool selection" in prompt


def test_intent_tool_context_lists_bound_tools_when_enabled():
    context = format_intent_tool_context([external_research_tools.search_web_intent]).lower()

    assert "intent agent tool catalog (callable by you now)" in context
    assert "only the tools in this section are callable by you" in context
    assert "`search_web_intent`" in context
    assert "intent web search" in context
    assert "do not use results as answer evidence" in context


def test_intent_tool_context_reports_no_active_tools_when_disabled():
    context = format_intent_tool_context([]).lower()

    assert "intent agent tool catalog (callable by you now)" in context
    assert "no intent-stage tools are active" in context


def test_arxiv_guidance_omits_dependency_version_detail():
    prompt = TOOL_FRIENDLY_CONFIG["arxiv"]["default_prompt"].lower()

    assert "arxiv==2.4.1" not in prompt
    assert "pinned" not in prompt
    assert "wrapper" not in prompt


def test_arxiv_dependency_matches_langchain_wrapper_api():
    arxiv = pytest.importorskip("arxiv")

    assert hasattr(arxiv.Search(query="test"), "results")


def test_orchestrator_tool_node_configures_recoverable_tool_errors():
    @tool
    def failing_tool(query: str) -> str:
        """Test tool that always fails."""
        raise RuntimeError("simulated tool outage")

    node = OrchestratorToolNode([failing_tool])
    message = node._handle_tool_errors(RuntimeError("simulated tool outage"))

    assert "Tool execution failed: RuntimeError: simulated tool outage" in message
    assert "continue with other available evidence" in message
