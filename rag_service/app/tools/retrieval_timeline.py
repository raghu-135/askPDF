"""Framework-neutral timeline retrieval."""

from datetime import datetime, timezone
from typing import Any

from app.agent.tool_contract import ToolWarningCode, make_tool_error_result, make_tool_result, tool_started
from app.rag.enums import ThreadTimelineOrder, ThreadTimelineSource, TimelineEventType, TimelineSourceType
from app.time_utils import parse_datetime_utc
from app.tools.contracts import TimelineRequest
from app.tools.context import ToolInvocationContext
from app.tools.services import DefaultToolServices, get_tool_services


def _excerpt(value: Any, limit: int = 260) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _sort_key(event: dict[str, Any], order: str) -> Any:
    parsed = parse_datetime_utc(event.get("timeline_event_at"))
    missing = parsed is None
    if order == ThreadTimelineOrder.OLDEST.value:
        return (missing, parsed or datetime.max.replace(tzinfo=timezone.utc))
    if order == ThreadTimelineOrder.NEWEST.value:
        return (missing, -(parsed or datetime.min.replace(tzinfo=timezone.utc)).timestamp())
    try:
        score = float(event.get("score") or 0.0)
    except Exception:
        score = 0.0
    return (-score, -(parsed.timestamp() if parsed else float("-inf")))


async def search_thread_events(request: TimelineRequest, context: ToolInvocationContext, *, services: DefaultToolServices | None = None):
    started = tool_started()
    tool_name = "search_thread_events"
    services = services or get_tool_services()
    try:
        if not context.thread_id or not context.embedding_model:
            return make_tool_result(tool_name=tool_name, content="No thread context found.", context=context, started=started, warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT])
        source = request.sources if request.sources in {item.value for item in ThreadTimelineSource} else ThreadTimelineSource.ALL.value
        order = request.order if request.order in {item.value for item in ThreadTimelineOrder} else ThreadTimelineOrder.RELEVANCE.value
        db = services.vector_db()
        vector = None
        if source in {ThreadTimelineSource.ALL.value, ThreadTimelineSource.CONVERSATION.value, ThreadTimelineSource.WEB_CACHE.value}:
            vector = await services.embed(context.embedding_model, request.query)
        events: list[dict[str, Any]] = []
        if vector is not None and source in {ThreadTimelineSource.ALL.value, ThreadTimelineSource.CONVERSATION.value}:
            recalled = await db.search_chat_memory(thread_id=context.thread_id, query_vector=vector, embedding_model=context.embedding_model, limit=request.max_results)
            if context.use_reranker and request.query:
                recalled = await services.rerank(request.query, recalled)
            for item in recalled:
                if item.get("message_created_at"):
                    events.append({"source_type": TimelineSourceType.CONVERSATION.value, "timeline_event_at": item["message_created_at"], "timeline_event_type": TimelineEventType.MESSAGE_CREATED.value, "message_created_at": item["message_created_at"], "message_id": item.get("message_id"), "label": "Conversation memory", "excerpt": _excerpt(item.get("text")), "score": item.get("rerank_score", item.get("score"))})
        if source in {ThreadTimelineSource.ALL.value, ThreadTimelineSource.DOCUMENTS.value}:
            lookup = await services.document_lookup(context.thread_id)
            for file_hash, meta in lookup.items():
                if not isinstance(meta, dict) or not meta.get("document_available_in_thread_at"):
                    continue
                events.append({"source_type": TimelineSourceType.DOCUMENT.value, "timeline_event_at": meta["document_available_in_thread_at"], "timeline_event_type": TimelineEventType.DOCUMENT_ADDED_TO_THREAD.value, "document_available_in_thread_at": meta["document_available_in_thread_at"], "file_hash": file_hash, "file_name": meta.get("file_name") or file_hash, "label": f"Document added to thread: {meta.get('file_name') or file_hash}", "excerpt": f"{meta.get('file_name') or file_hash} was added to this thread."})
        if vector is not None and source in {ThreadTimelineSource.ALL.value, ThreadTimelineSource.WEB_CACHE.value}:
            chunks = await db.search_web_chunks(thread_id=context.thread_id, query_vector=vector, embedding_model=context.embedding_model, limit=request.max_results, query_text=request.query)
            if context.use_reranker and request.query:
                chunks = await services.rerank(request.query, chunks)
            for item in chunks:
                if item.get("web_search_performed_at"):
                    events.append({"source_type": TimelineSourceType.WEB_CACHE.value, "timeline_event_at": item["web_search_performed_at"], "timeline_event_type": TimelineEventType.WEB_SEARCH_PERFORMED.value, "web_search_performed_at": item["web_search_performed_at"], "url": item.get("url", ""), "title": item.get("title") or "Internet Search", "label": f"Cached web result: \"{item.get('title') or 'Internet Search'}\"", "excerpt": _excerpt(item.get("text")), "score": item.get("rerank_score", item.get("score"))})
        events = sorted(events, key=lambda item: _sort_key(item, order))[:request.max_results]
        from app.agent.evidence_contract import evidence_segment
        return make_tool_result(tool_name=tool_name, content="No timeline events matched the request." if not events else "[THREAD TIMELINE EVENTS]\n" + "\n".join(f"- {item.get('timeline_event_at') or 'unknown time'} | {item.get('timeline_event_type') or 'unknown_event'} | {item.get('label') or item.get('source_type')}: {item.get('excerpt') or ''}" for item in events), context=context, started=started, sources=events, artifacts={"timeline_events": events, "evidence_segments": [s for event in events if (s := evidence_segment(kind="timeline", content=event.get("excerpt"), source=event, raw_score=event.get("score")))]}, warnings=[] if events else [ToolWarningCode.NO_TIMELINE_EVENTS])
    except Exception as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=f"Error searching thread timeline: {exc}")
