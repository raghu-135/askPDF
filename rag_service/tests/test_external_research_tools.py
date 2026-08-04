import pytest
from langchain_core.tools import tool

from app.agent import external_research_tools
from app.agent.tool_contract import normalize_tool_result
from app.agent.tool_node import RecoverableToolNode
from app.prompts.loaders import get_web_search_mandate
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
        "search_thread_events",
    }

    missing = expected_tool_names - set(TOOL_FRIENDLY_CONFIG)
    assert not missing
    assert "find_topic_anchor_in_history" not in TOOL_FRIENDLY_CONFIG


@pytest.mark.asyncio
async def test_external_tool_wrapper_returns_contract_envelope():
    @tool
    async def fake_reference(query: str) -> str:
        """Lookup fake reference material."""
        return f"reference: {query}"

    wrapped = external_research_tools._wrap_external_tool_with_contract(fake_reference)
    raw = await wrapped.ainvoke(
        {"query": "diffusion"},
        config={"configurable": {"agent_run_id": "run-1", "caller_node": "web_worker"}},
    )
    payload = normalize_tool_result(raw, tool_name=wrapped.name)

    assert wrapped.name == "fake_reference"
    assert payload["ok"] is True
    assert payload["content"] == "reference: diffusion"
    assert payload["trace"]["agent_run_id"] == "run-1"
    assert payload["trace"]["caller_node"] == "web_worker"
    assert payload["artifacts"]["provider_tool"]


@pytest.mark.asyncio
async def test_external_tool_wrapper_converts_provider_errors_to_recoverable_result():
    @tool
    async def failing_reference(query: str) -> str:
        """Lookup fake reference material."""
        raise RuntimeError("provider unavailable")

    wrapped = external_research_tools._wrap_external_tool_with_contract(failing_reference)
    raw = await wrapped.ainvoke({"query": "diffusion"})
    payload = normalize_tool_result(raw, tool_name=wrapped.name)

    assert payload["ok"] is False
    assert payload["error"]["type"] == "RuntimeError"
    assert "provider unavailable" in payload["content"]


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

    node = RecoverableToolNode([failing_tool])
    message = node._handle_tool_errors(RuntimeError("simulated tool outage"))

    assert "Tool execution failed: RuntimeError: simulated tool outage" in message
    assert "continue with other available evidence" in message
