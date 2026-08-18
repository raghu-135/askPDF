"""MCP caller adapters for web and external research tools.

This module intentionally contains no provider implementations.  It remains
as a small import-compatible surface for callers that still obtain tools from
the historical module; every returned tool is an MCP client adapter.
"""

from typing import Any

from langchain_core.tools import BaseTool

from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG
from app.mcp.langchain_adapter import create_mcp_langchain_tool
from app.rag.enums import TimelineEventType


search_web = create_mcp_langchain_tool("search_web")


def _format_web_context(
    texts: list[str],
    urls: list[str],
    titles: list[str],
    scores: list[float] | None = None,
    web_search_performed_at: str | None = None,
) -> dict[str, Any]:
    """Format web evidence for old presentation callers without executing tools."""
    web_groups: dict[str, dict[str, Any]] = {}
    web_sources: list[dict[str, Any]] = []
    for idx, (text, url, title) in enumerate(zip(texts, urls, titles)):
        if url not in web_groups:
            web_groups[url] = {"title": title or url or "Internet Search", "texts": []}
        web_groups[url]["texts"].append(text)
        entry: dict[str, Any] = {"text": text[:200] + "...", "url": url, "title": title or "Internet Search"}
        if scores and idx < len(scores):
            entry["score"] = scores[idx]
        if web_search_performed_at:
            entry.update({
                "web_search_performed_at": web_search_performed_at,
                "timeline_event_at": web_search_performed_at,
                "timeline_event_type": TimelineEventType.WEB_SEARCH_PERFORMED.value,
            })
        web_sources.append(entry)

    content = []
    for url, group in web_groups.items():
        prefix = f"Web result from search performed at {web_search_performed_at}:\n" if web_search_performed_at else ""
        content.append(f'{prefix}[Source: Internet Search — "{group["title"]}" | {url}]\n' + "\n".join(group["texts"]))
    return {"content": "\n\n".join(content), "__web_sources__": web_sources}


def get_external_research_tools() -> list[BaseTool]:
    """Return MCP adapters for all registry-authorized external tools."""
    names = (
        "wikipedia",
        "wikidata",
        "arxiv",
        "pub_med",
        "pubmed",
        "semanticscholar",
        "semantic_scholar",
        "stack_exchange",
        "yahoo_finance_news",
    )
    return [create_mcp_langchain_tool(name) for name in names if name in TOOL_FRIENDLY_CONFIG]
