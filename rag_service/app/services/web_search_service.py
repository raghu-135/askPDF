"""Framework-neutral live web search without workflow or persistence concerns."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Dict, List

from ddgs import DDGS

from app.rag.retrieval import rerank_document_chunks
from app.services.retry import run_with_bounded_retries
from app.time_utils import iso_utc_z


WEB_SEARCH_CAPABILITY = "web:search"
DEFAULT_WEB_SEARCH_RESULTS = 6


def _search_provider(query: str, max_results: int) -> List[Dict[str, Any]]:
    """Call the supported DDGS package behind an injectable sync boundary."""

    return list(DDGS().text(query, max_results=max_results))


def _normalize_results(raw: Any) -> List[Dict[str, str]]:
    if isinstance(raw, list):
        return raw
    return []


async def search_internet(
    query: str,
    *,
    max_results: int = DEFAULT_WEB_SEARCH_RESULTS,
    use_reranker: bool = True,
) -> Dict[str, Any]:
    """Return bounded structured live-search evidence; never persist it."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("Web search query cannot be empty")
    if max_results <= 0:
        raise ValueError("max_results must be positive")
    raw = await run_with_bounded_retries(
        lambda: asyncio.to_thread(_search_provider, normalized_query, max_results),
        operation_name="web search",
    )
    rows = _normalize_results(raw)
    chunks = [
        {
            "text": str(row.get("snippet", row.get("body", ""))).strip(),
            "url": str(row.get("link", row.get("href", ""))).strip(),
            "title": str(row.get("title", "")).strip(),
        }
        for row in rows
        if str(row.get("snippet", row.get("body", ""))).strip()
    ][:max_results]
    if use_reranker and chunks:
        chunks = await rerank_document_chunks(normalized_query, chunks)
    searched_at = iso_utc_z()
    sources = []
    for row in chunks[:max_results]:
        url = str(row.get("url") or "")
        title = str(row.get("title") or url or "Internet Search")
        source_id = hashlib.sha256(
            f"{normalized_query}\0{url}\0{title}".encode("utf-8")
        ).hexdigest()[:24]
        sources.append({
            "id": source_id,
            "title": title,
            "url": url,
            "snippet": str(row.get("text") or "")[:1000],
            "query": normalized_query,
            "searched_at": searched_at,
            **({"score": row["rerank_score"]} if row.get("rerank_score") is not None else {}),
        })
    return {"query": normalized_query, "searched_at": searched_at, "sources": sources}


def format_search_context(result: Dict[str, Any]) -> str:
    parts = []
    for source in result.get("sources") or []:
        parts.append(
            f'[Web source {source["id"]}: "{source["title"]}" | {source["url"]}]\n'
            f'{source["snippet"]}'
        )
    return "\n\n".join(parts)
