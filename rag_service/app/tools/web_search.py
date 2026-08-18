"""Framework-neutral live web search handler."""

import asyncio

from app.agent.tool_contract import ToolWarningCode, make_tool_error_result, make_tool_result, tool_started
from app.agent.evidence_contract import evidence_segment
from app.services.web_search_service import DEFAULT_WEB_SEARCH_RESULTS, search_internet
from app.time_utils import iso_utc_z
from app.tools.contracts import QueryRequest
from app.tools.context import ToolInvocationContext
from app.tools.services import DefaultToolServices, get_tool_services
from app.tools.background_tasks import register_background_task


async def search_web(request: QueryRequest, context: ToolInvocationContext, *, services: DefaultToolServices | None = None):
    started = tool_started()
    tool_name = "search_web"
    services = services or get_tool_services()
    try:
        if not context.use_web_search:
            return make_tool_result(tool_name=tool_name, content="Internet search is not enabled for this session. The user has not turned on web search, so no internet results are available. Answer using only the uploaded documents and conversation history.", context=context, started=started, warnings=[ToolWarningCode.WEB_SEARCH_DISABLED])
        result = await search_internet(request.query, max_results=DEFAULT_WEB_SEARCH_RESULTS, use_reranker=False)
        sources = result.get("sources") or [] if isinstance(result, dict) else []
        if not sources:
            return make_tool_result(tool_name=tool_name, content="Web search returned no usable text.", context=context, started=started, warnings=[ToolWarningCode.NO_USABLE_WEB_RESULTS])
        performed_at = iso_utc_z()
        chunks = [{"text": item.get("snippet", ""), "url": item.get("url", ""), "title": item.get("title", "")} for item in sources]
        scores = None
        if context.use_reranker:
            chunks = await services.rerank(request.query, chunks)
            scores = [item.get("rerank_score") for item in chunks]
        texts = [item.get("text", "") for item in chunks]
        urls = [item.get("url", "") for item in chunks]
        titles = [item.get("title", "") for item in chunks]
        if context.thread_id and context.embedding_model and context.web_search_index:
            from app.rag.indexer import index_web_search_for_thread
            register_background_task(
                context.cancellation_scope_id,
                asyncio.create_task(index_web_search_for_thread(thread_id=context.thread_id, query=request.query, texts=texts, urls=urls, titles=titles, embedding_model=context.embedding_model, web_search_performed_at=performed_at)),
            )
        web_sources = [{"text": text[:200] + "...", "url": urls[i], "title": titles[i], **({"score": scores[i]} if scores and i < len(scores) else {}), "web_search_performed_at": performed_at, "timeline_event_at": performed_at} for i, text in enumerate(texts)]
        content = "\n\n".join(f'[Source: Internet Search — "{titles[i] or urls[i]}" | {urls[i]}]\n{text}' for i, text in enumerate(texts))
        return make_tool_result(tool_name=tool_name, content=content, context=context, started=started, sources=web_sources, artifacts={"web_sources": web_sources, "evidence_segments": [s for i, text in enumerate(texts) if (s := evidence_segment(kind="web", content=text, source={"url": urls[i], "title": titles[i], "web_search_performed_at": performed_at}, raw_score=scores[i] if scores and i < len(scores) else None))]})
    except Exception as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=f"Web search failed: {exc}")
