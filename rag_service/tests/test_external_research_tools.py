import os
from pathlib import Path

import pytest
from langchain_core.tools import tool

from app.agent import external_research_tools
from app.agent.tool_contract import normalize_tool_result
from langgraph_runtime.agent.tool_node import RecoverableToolNode
from app.prompts.loaders import get_web_search_mandate
from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG


TOOL_PACKAGE_PINS = {
    "langgraph": "1.2.6",
    "langchain-core": "1.4.8",
    "ddgs": "9.14.4",
}


def _requirements_lines() -> set[str]:
    repository_root = Path(os.getenv("ASKPDF_REPO_DIR", Path(__file__).resolve().parents[2]))
    lines: set[str] = set()
    for path in (
        repository_root / "rag_service/requirements-control-plane.txt",
        repository_root / "langgraph_runtime/requirements.txt",
    ):
        with path.open(encoding="utf-8") as req_file:
            lines.update(
                line.strip()
                for line in req_file
                if line.strip() and not line.lstrip().startswith("#") and not line.startswith("-r ")
            )
    return lines


def test_tool_dependencies_are_exactly_pinned():
    requirements = _requirements_lines()
    missing = {
        f"{package}=={version}"
        for package, version in TOOL_PACKAGE_PINS.items()
        if f"{package}=={version}" not in requirements
    }

    assert not missing


def test_external_research_tool_candidates_exclude_searxng(monkeypatch):
    """External provider discovery returns MCP adapters, never provider tools."""
    tools = external_research_tools.get_external_research_tools()
    assert {item.name for item in tools} >= {
        "wikipedia", "wikidata", "arxiv", "pub_med", "semanticscholar",
        "stack_exchange", "yahoo_finance_news",
    }
    assert all(not hasattr(item, "provider") for item in tools)


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
