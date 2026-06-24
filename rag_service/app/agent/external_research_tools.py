"""
Externally backed LangChain Community research tools.

The built-in web search tools are always defined here. Additional source
specific tools are registered only when their package/runtime dependencies are
available, so one broken provider cannot prevent the agent from starting.
"""

import importlib
import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import BaseTool, tool
from langchain_core.runnables import RunnableConfig

from app.rag.retrieval import rerank_document_chunks

logger = logging.getLogger(__name__)

search_tool = DuckDuckGoSearchResults(output_format="list", num_results=6)


def _import_attr(module_path: str, attr_name: str) -> Any:
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)


def _normalize_web_results(raw: Any, query: str) -> List[Dict[str, str]]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return [{"snippet": raw, "title": query, "link": ""}]
    return []


async def _run_web_search(query: str, max_results: Optional[int]) -> Optional[Dict[str, List[str]]]:
    raw = await asyncio.to_thread(search_tool.invoke, query)
    logger.info(f"Raw search results for '{query}': {raw}")
    results_list = _normalize_web_results(raw, query)
    if not results_list:
        return None

    texts = [r.get("snippet", r.get("body", "")) for r in results_list]
    urls = [r.get("link", r.get("href", "")) for r in results_list]
    titles = [r.get("title", "") for r in results_list]

    valid = [(t, u, ti) for t, u, ti in zip(texts, urls, titles) if t.strip()]
    if not valid:
        return None

    texts, urls, titles = zip(*valid)
    texts = list(texts)
    urls = list(urls)
    titles = list(titles)
    if max_results is not None:
        texts = texts[:max_results]
        urls = urls[:max_results]
        titles = titles[:max_results]
    return {"texts": texts, "urls": urls, "titles": titles}


def _format_web_context(
    texts: List[str],
    urls: List[str],
    titles: List[str],
    scores: Optional[List[float]] = None,
) -> Dict[str, Any]:
    web_groups: Dict[str, Dict[str, Any]] = {}
    web_sources: List[Dict[str, Any]] = []
    for idx, (text, url, title) in enumerate(zip(texts, urls, titles)):
        if url not in web_groups:
            web_groups[url] = {"title": title or url or "Internet Search", "texts": []}
        web_groups[url]["texts"].append(text)
        entry: Dict[str, Any] = {
            "text": text[:200] + "...",
            "url": url,
            "title": title or "Internet Search",
        }
        if scores and idx < len(scores):
            entry["score"] = scores[idx]
        web_sources.append(entry)

    context_parts = []
    for url, group in web_groups.items():
        combined_text = "\n".join(group["texts"])
        context_parts.append(f'[Source: Internet Search — "{group["title"]}" | {url}]\n{combined_text}')

    return {
        "content": "\n\n".join(context_parts),
        "__web_sources__": web_sources,
    }


@tool
async def search_web(query: str, config: RunnableConfig = None) -> str:
    """
    Live web search for external or time-sensitive information.
    Results are cached to the thread and returned with labeled sources.

    Args:
        query: Concise, keyword-rich search query.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        if not conf.get("use_web_search", False):
            return "Internet search is not enabled for this session. The user has not turned on web search, so no internet results are available. Answer using only the uploaded documents and conversation history."
        use_reranker = conf.get("use_reranker", True)

        logger.info(f"--- WEB SEARCH INITIATED --- Query: '{query}'")
        thread_id = conf.get("thread_id")
        embedding_model = conf.get("embedding_model")

        result = await _run_web_search(query, max_results=6)
        if not result:
            return "Web search returned no usable text."

        texts = result["texts"]
        urls = result["urls"]
        titles = result["titles"]
        scores: Optional[List[float]] = None
        if use_reranker:
            web_chunks = [{"text": t, "url": urls[i], "title": titles[i]} for i, t in enumerate(texts)]
            web_chunks = await rerank_document_chunks(query, web_chunks)
            texts = [c.get("text", "") for c in web_chunks]
            urls = [c.get("url", "") for c in web_chunks]
            titles = [c.get("title", "") for c in web_chunks]
            scores = [c.get("rerank_score") for c in web_chunks]

        # Persist results in Weaviate for future retrieval.
        if thread_id and embedding_model and conf.get("web_search_index", True):
            try:
                from app.rag.indexer import index_web_search_for_thread
                asyncio.create_task(
                    index_web_search_for_thread(
                        thread_id=thread_id,
                        query=query,
                        texts=texts,
                        urls=urls,
                        titles=titles,
                        embedding_model_name=embedding_model,
                    )
                )
            except Exception as idx_err:
                logger.warning(f"Web search indexing skipped: {idx_err}")

        return json.dumps(_format_web_context(texts, urls, titles, scores=scores))
    except Exception as e:
        logger.error(f"Web search failed: {e}", exc_info=True)
        return f"Web search failed: {str(e)}"


@tool
async def search_web_intent(query: str, config: RunnableConfig = None) -> str:
    """
    Lightweight web lookup for intent disambiguation and time-sensitivity checks.
    Use for query rewriting only; do not cite as evidence.

    Args:
        query: Concise query aimed at identifying a term or entity.
    """
    try:
        conf = config.get("configurable", {}) if config else {}
        if not conf.get("use_web_search", False):
            return "Internet search is not enabled for this session. The user has not turned on web search, so no internet results are available."

        logger.info(f"--- INTENT WEB SEARCH INITIATED --- Query: '{query}'")
        result = await _run_web_search(query, max_results=None)
        if not result:
            return "Web search returned no usable text."

        return json.dumps(_format_web_context(result["texts"], result["urls"], result["titles"]))
    except Exception as e:
        logger.error(f"Intent web search failed: {e}", exc_info=True)
        return f"Web search failed: {str(e)}"


def _build_tool(
    display_name: str,
    tool_path: str,
    class_name: str,
    factory: Optional[Callable[[Any], BaseTool]] = None,
) -> Optional[BaseTool]:
    try:
        tool_cls = _import_attr(tool_path, class_name)
        tool = factory(tool_cls) if factory else tool_cls()
        logger.info("Registered external research tool: %s (%s)", display_name, tool.name)
        return tool
    except Exception as exc:
        logger.warning("Skipping external research tool %s: %s", display_name, exc)
        return None


def _wikipedia_tool(tool_cls: Any) -> BaseTool:
    wrapper_cls = _import_attr(
        "langchain_community.utilities.wikipedia",
        "WikipediaAPIWrapper",
    )
    return tool_cls(
        api_wrapper=wrapper_cls(top_k_results=3, doc_content_chars_max=3000)
    )


def _arxiv_tool(tool_cls: Any) -> BaseTool:
    wrapper_cls = _import_attr(
        "langchain_community.utilities.arxiv",
        "ArxivAPIWrapper",
    )
    return tool_cls(
        api_wrapper=wrapper_cls(top_k_results=3, doc_content_chars_max=3000)
    )


def _pubmed_tool(tool_cls: Any) -> BaseTool:
    wrapper_cls = _import_attr(
        "langchain_community.utilities.pubmed",
        "PubMedAPIWrapper",
    )
    return tool_cls(
        api_wrapper=wrapper_cls(top_k_results=3, doc_content_chars_max=3000)
    )


def _semantic_scholar_tool(tool_cls: Any) -> BaseTool:
    wrapper_cls = _import_attr(
        "langchain_community.utilities.semanticscholar",
        "SemanticScholarAPIWrapper",
    )
    return tool_cls(api_wrapper=wrapper_cls(top_k_results=3))


def _wikidata_tool(_: Any) -> BaseTool:
    @tool
    def wikidata(query: str) -> str:
        """
        Lookup structured entity facts from Wikidata.

        Args:
            query: Entity or fact lookup query.
        """
        tool_cls = _import_attr(
            "langchain_community.tools.wikidata.tool",
            "WikidataQueryRun",
        )
        wrapper_cls = _import_attr(
            "langchain_community.utilities.wikidata",
            "WikidataAPIWrapper",
        )
        wikidata_tool = tool_cls(api_wrapper=wrapper_cls())
        return wikidata_tool.invoke(query)

    return wikidata


def _stackexchange_tool(_: Any) -> BaseTool:
    @tool
    def stack_exchange(query: str) -> str:
        """
        Search Stack Exchange / Stack Overflow style technical Q&A.

        Args:
            query: Technical search query.
        """
        tool_cls = _import_attr(
            "langchain_community.tools.stackexchange.tool",
            "StackExchangeTool",
        )
        wrapper_cls = _import_attr(
            "langchain_community.utilities.stackexchange",
            "StackExchangeAPIWrapper",
        )
        stack_tool = tool_cls(api_wrapper=wrapper_cls(max_results=3))
        return stack_tool.invoke(query)

    return stack_exchange


def get_external_research_tools() -> List[BaseTool]:
    """
    Return enabled free/public LangChain Community research tools.

    SearXNG tools are intentionally excluded because they require a configured
    self-hosted SearXNG instance.
    """
    candidates = [
        (
            "Wikipedia",
            "langchain_community.tools.wikipedia.tool",
            "WikipediaQueryRun",
            _wikipedia_tool,
        ),
        (
            "Wikidata",
            "langchain_community.tools.wikidata.tool",
            "WikidataQueryRun",
            _wikidata_tool,
        ),
        (
            "arXiv",
            "langchain_community.tools.arxiv.tool",
            "ArxivQueryRun",
            _arxiv_tool,
        ),
        (
            "PubMed",
            "langchain_community.tools.pubmed.tool",
            "PubmedQueryRun",
            _pubmed_tool,
        ),
        (
            "Semantic Scholar",
            "langchain_community.tools.semanticscholar.tool",
            "SemanticScholarQueryRun",
            _semantic_scholar_tool,
        ),
        (
            "StackExchange",
            "langchain_community.tools.stackexchange.tool",
            "StackExchangeTool",
            _stackexchange_tool,
        ),
        (
            "Yahoo Finance News",
            "langchain_community.tools.yahoo_finance_news",
            "YahooFinanceNewsTool",
            lambda tool_cls: tool_cls(top_k=5),
        ),
    ]

    tools: List[BaseTool] = []
    for display_name, module_path, class_name, factory in candidates:
        tool = _build_tool(display_name, module_path, class_name, factory)
        if tool is not None:
            tools.append(tool)
    return tools
