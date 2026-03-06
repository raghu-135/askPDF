import asyncio
import json
import logging
from typing import Any, Dict, List, Tuple

from langchain_community.tools import DuckDuckGoSearchResults

search_tool = DuckDuckGoSearchResults(output_format="list", num_results=6)


def _should_prefetch_web(question: str) -> bool:
    q = (question or "").lower()
    if len(q.split()) > 6:
        return True
    if any(w in q for w in ("what", "when", "where", "who", "why", "how")):
        return True
    if any(ch.isdigit() for ch in q):
        return True
    if "latest" in q or "current" in q:
        return True
    return False


async def prefetch_web_context(
    raw_question: str,
    thread_id: str,
    embed_model_name: str,
    use_web_search: bool,
    reasoning_mode: bool,
    logger: logging.Logger,
) -> Tuple[str, List[Dict[str, Any]]]:
    if not use_web_search or reasoning_mode or not _should_prefetch_web(raw_question):
        return "", []

    try:
        logger.info("Prefetch web search triggered for non-reasoning mode.")
        raw = await asyncio.to_thread(search_tool.invoke, raw_question)
        if isinstance(raw, list):
            results_list = raw
        elif isinstance(raw, str):
            try:
                results_list = json.loads(raw)
            except Exception:
                results_list = [{"snippet": raw, "title": raw_question, "link": ""}]
        else:
            results_list = []

        if not results_list:
            return "", []

        texts = [r.get("snippet", r.get("body", "")) for r in results_list]
        urls = [r.get("link", r.get("href", "")) for r in results_list]
        titles = [r.get("title", "") for r in results_list]
        valid = [(t, u, ti) for t, u, ti in zip(texts, urls, titles) if t.strip()]
        if not valid:
            return "", []
        texts, urls, titles = zip(*valid)
        texts = list(texts)
        urls = list(urls)
        titles = list(titles)

        try:
            from rag import index_web_search_for_thread
            asyncio.create_task(
                index_web_search_for_thread(
                    thread_id=thread_id,
                    query=raw_question,
                    texts=texts,
                    urls=urls,
                    titles=titles,
                    embedding_model_name=embed_model_name,
                )
            )
        except Exception as idx_err:
            logger.warning(f"Web prefetch indexing skipped: {idx_err}")

        web_sources = []
        groups = {}
        for text, url, title in zip(texts, urls, titles):
            if url not in groups:
                groups[url] = {"title": title or url or "Internet Search", "texts": []}
            groups[url]["texts"].append(text)
            web_sources.append({
                "text": text[:200] + "...",
                "url": url,
                "title": title or "Internet Search",
            })

        context_parts = []
        for url, group in groups.items():
            combined_text = "\n".join(group["texts"])
            context_parts.append(f'[Source: Internet Search — "{group["title"]}" | {url}]\n{combined_text}')

        return "\n\n".join(context_parts), web_sources
    except Exception as exc:
        logger.warning(f"Prefetch web search failed: {exc}")
        return "", []
