"""Lazy, dependency-isolated provider clients.

Each provider is constructed only when selected.  Provider failures therefore
cannot prevent rag-service startup or hide unrelated tools.
"""

import xml.etree.ElementTree as ET
import re
from typing import Any

from app.http_clients import get_http_client
from app.tools.context import ToolInvocationContext
from app.tools.provider_ports import ProviderResult


async def _json(url: str, *, params: dict[str, Any]) -> dict[str, Any]:
    response = await get_http_client("providers").get(url, params=params, headers={"User-Agent": "askPDF/1.0"})
    response.raise_for_status()
    if len(response.content) > 5_000_000:
        raise ValueError("provider response exceeded size limit")
    value = response.json()
    return value if isinstance(value, dict) else {}


class WikipediaProvider:
    name = "wikipedia"

    async def search(self, query: str, *, context: ToolInvocationContext) -> ProviderResult:
        del context
        search = await _json("https://en.wikipedia.org/w/api.php", params={"action": "query", "list": "search", "srsearch": query, "srlimit": 3, "format": "json", "formatversion": 2})
        titles = [item.get("title") for item in (search.get("query", {}).get("search", []) or []) if isinstance(item, dict) and item.get("title")]
        parts, sources = [], []
        for title in titles:
            page = await _json("https://en.wikipedia.org/w/api.php", params={"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1, "redirects": 1, "titles": title, "exchars": 3000, "format": "json", "formatversion": 2})
            item = (page.get("query", {}).get("pages", []) or [{}])[0]
            summary = str(item.get("extract") or "").strip()
            if summary:
                parts.append(f"Page: {item.get('title') or title}\nSummary: {summary[:3000]}")
                sources.append({"title": item.get("title") or title, "url": f"https://en.wikipedia.org/wiki/{str(title).replace(' ', '_')}"})
        return ProviderResult("\n\n".join(parts) or "No good Wikipedia Search Result was found", sources)


class WikidataProvider:
    name = "wikidata"

    async def search(self, query: str, *, context: ToolInvocationContext) -> ProviderResult:
        del context
        data = await _json("https://www.wikidata.org/w/api.php", params={"action": "wbsearchentities", "search": query, "language": "en", "format": "json", "limit": 5})
        items = data.get("search", []) or []
        lines = [f"{item.get('label') or item.get('id')}: {item.get('description') or ''}" for item in items if isinstance(item, dict)]
        return ProviderResult("\n".join(lines) or "No Wikidata results found.", [{"id": item.get("id"), "title": item.get("label")} for item in items if isinstance(item, dict)])


class ArxivProvider:
    name = "arxiv"

    async def search(self, query: str, *, context: ToolInvocationContext) -> ProviderResult:
        del context
        response = await get_http_client("providers").get("https://export.arxiv.org/api/query", params={"search_query": f"all:{query}", "max_results": 5}, headers={"User-Agent": "askPDF/1.0"})
        response.raise_for_status()
        root = ET.fromstring(response.text[:5_000_000])
        ns = {"a": "http://www.w3.org/2005/Atom"}
        parts, sources = [], []
        for entry in root.findall("a:entry", ns):
            title = " ".join((entry.findtext("a:title", "", ns) or "").split())
            summary = " ".join((entry.findtext("a:summary", "", ns) or "").split())
            link = entry.findtext("a:id", "", ns)
            if title:
                parts.append(f"Title: {title}\nAbstract: {summary}")
                sources.append({"title": title, "url": link})
        return ProviderResult("\n\n".join(parts) or "No arXiv results found.", sources)


class PubMedProvider:
    name = "pub_med"

    async def search(self, query: str, *, context: ToolInvocationContext) -> ProviderResult:
        del context
        found = await _json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 5})
        ids = (found.get("esearchresult", {}) or {}).get("idlist", [])
        if not ids:
            return ProviderResult("No PubMed results found.")
        summary = await _json("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi", params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
        result = summary.get("result", {})
        lines, sources = [], []
        for item_id in ids:
            item = result.get(item_id, {})
            title = item.get("title") or item_id
            lines.append(f"Title: {title}\nJournal: {item.get('fulljournalname') or item.get('source') or ''}")
            sources.append({"title": title, "url": f"https://pubmed.ncbi.nlm.nih.gov/{item_id}/", "pmid": item_id})
        return ProviderResult("\n\n".join(lines), sources)


class SemanticScholarProvider:
    name = "semanticscholar"

    async def search(self, query: str, *, context: ToolInvocationContext) -> ProviderResult:
        del context
        data = await _json("https://api.semanticscholar.org/graph/v1/paper/search", params={"query": query, "limit": 5, "fields": "title,abstract,year,url"})
        papers = data.get("data", []) or []
        lines = [f"Title: {p.get('title')} ({p.get('year') or 'n.d.'})\nAbstract: {p.get('abstract') or ''}" for p in papers]
        return ProviderResult("\n\n".join(lines) or "No Semantic Scholar results found.", [{"title": p.get("title"), "url": p.get("url")} for p in papers])


class StackExchangeProvider:
    name = "stack_exchange"

    async def search(self, query: str, *, context: ToolInvocationContext) -> ProviderResult:
        del context
        data = await _json("https://api.stackexchange.com/2.3/search/advanced", params={"order": "desc", "sort": "relevance", "q": query, "site": "stackoverflow", "pagesize": 5})
        items = data.get("items", []) or []
        lines = [f"{item.get('title')}\n{item.get('link')}" for item in items]
        return ProviderResult("\n\n".join(lines) or "No StackExchange results found.", [{"title": i.get("title"), "url": i.get("link")} for i in items])


class YahooFinanceNewsProvider:
    name = "yahoo_finance_news"

    async def search(self, query: str, *, context: ToolInvocationContext) -> ProviderResult:
        del context
        ticker = query.strip().upper()
        if not re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z]{1,2})?", ticker):
            raise ValueError("Yahoo Finance News requires a valid ticker symbol")
        data = await _json("https://query1.finance.yahoo.com/v1/finance/search", params={"q": ticker, "newsCount": 5})
        news = data.get("news", []) or []
        lines = [f"{item.get('title')}\n{item.get('publisher') or ''}\n{item.get('link') or ''}" for item in news]
        return ProviderResult("\n\n".join(lines) or "No Yahoo Finance News results found.", [{"title": i.get("title"), "url": i.get("link")} for i in news])


PROVIDERS = {
    "wikipedia": WikipediaProvider,
    "wikidata": WikidataProvider,
    "arxiv": ArxivProvider,
    "pub_med": PubMedProvider,
    "pubmed": PubMedProvider,
    "semanticscholar": SemanticScholarProvider,
    "semantic_scholar": SemanticScholarProvider,
    "stack_exchange": StackExchangeProvider,
    "yahoo_finance_news": YahooFinanceNewsProvider,
}
