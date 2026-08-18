"""Framework-neutral document retrieval handlers."""

from typing import Any

from app.agent.tool_contract import ToolWarningCode, make_tool_error_result, make_tool_result, tool_started
from app.tools.contracts import DocumentSearchRequest, FocusedDocumentSearchRequest
from app.tools.context import ToolInvocationContext
from app.tools.services import DefaultToolServices, get_tool_services


async def search_documents(
    request: DocumentSearchRequest,
    context: ToolInvocationContext,
    *,
    services: DefaultToolServices | None = None,
):
    started = tool_started()
    tool_name = "search_documents"
    services = services or get_tool_services()
    try:
        if not context.thread_id or not context.embedding_model:
            return make_tool_result(tool_name=tool_name, content="No thread context found.", context=context, started=started, warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT])
        query_vector = await services.embed(context.embedding_model, request.query)
        lookup = await services.document_lookup(context.thread_id)
        file_hashes = list(lookup.keys())
        if not file_hashes:
            return make_tool_result(tool_name=tool_name, content="No documents are linked to this thread yet.", context=context, started=started, warnings=[ToolWarningCode.NO_THREAD_DOCUMENTS])
        db = services.vector_db()
        raw = await db.search_knowledge_sources(thread_id=context.thread_id, query_vector=query_vector, embedding_model=context.embedding_model, limit=request.max_results, file_hashes=file_hashes, query_text=request.query)
        if not raw:
            return make_tool_result(tool_name=tool_name, content="Document index is missing for this thread. Re-open the thread to trigger re-indexing.", context=context, started=started, warnings=[ToolWarningCode.MISSING_DOCUMENT_VECTORS])
        if context.use_reranker:
            raw = await services.rerank(request.query, raw)
        radius = max(2, min(10, int((context.context_window or 32000) / 8000) + 1))
        by_file: dict[str, set[int]] = {}
        for hit in raw:
            file_hash, chunk_id = hit.get("file_hash"), hit.get("chunk_id")
            if file_hash is None or chunk_id is None:
                continue
            by_file.setdefault(file_hash, set()).update(i for i in range(int(chunk_id) - radius, int(chunk_id) + radius + 1) if i >= 0)
        expanded: list[dict[str, Any]] = []
        for file_hash, ids in by_file.items():
            expanded.extend(await db.get_knowledge_source_chunks_by_ids(thread_id=context.thread_id, embedding_model=context.embedding_model, file_hash=file_hash, chunk_ids=list(ids)))
        expanded.sort(key=lambda item: (item.get("file_hash", ""), item.get("chunk_id", 0)))
        web = await db.search_web_chunks(thread_id=context.thread_id, query_vector=query_vector, embedding_model=context.embedding_model, limit=max(3, request.max_results // 3), query_text=request.query)
        if context.use_reranker:
            web = await services.rerank(request.query, web)
        if not expanded and not web:
            return make_tool_result(tool_name=tool_name, content="No relevant content found in documents or cached web results.", context=context, started=started, warnings=[ToolWarningCode.NO_RELEVANT_CONTENT])
        from app.rag.retrieval import group_document_chunks
        document_content, document_sources = group_document_chunks(expanded, lookup)
        web_sources: list[dict[str, Any]] = []
        groups: dict[str, dict[str, Any]] = {}
        for chunk in web:
            url = chunk.get("url", "")
            groups.setdefault(url, {"title": chunk.get("title", url), "texts": [], "web_search_performed_at": chunk.get("web_search_performed_at")})["texts"].append(chunk.get("text", ""))
            item = {"text": chunk.get("text", "")[:200] + "...", "url": url, "title": chunk.get("title", url), "score": chunk.get("rerank_score", chunk.get("score", 0.0))}
            for field in ("web_search_performed_at", "timeline_event_at", "timeline_event_type"):
                if chunk.get(field) not in (None, ""):
                    item[field] = chunk[field]
            web_sources.append(item)
        parts = [document_content] if document_content else []
        for url, group in groups.items():
            prefix = f"Cached web result from search performed at {group['web_search_performed_at']}:\n" if group.get("web_search_performed_at") else ""
            parts.append(f'{prefix}[Source: Internet Search - "{group["title"]}" | {url}]\n' + "\n".join(group["texts"]))
        from app.agent.evidence_contract import evidence_segment
        segments = [s for chunk in expanded if (s := evidence_segment(kind="document", content=chunk.get("text"), source=chunk, raw_score=chunk.get("rerank_score", chunk.get("score"))))]
        segments += [s for chunk in web if (s := evidence_segment(kind="web", content=chunk.get("text"), source=chunk, raw_score=chunk.get("rerank_score", chunk.get("score"))))]
        return make_tool_result(tool_name=tool_name, content="\n\n".join(parts), context=context, started=started, sources=[*document_sources, *web_sources], artifacts={"document_sources": document_sources, "web_sources": web_sources, "evidence_segments": segments})
    except Exception as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=f"Error retrieving knowledge: {exc}")


async def search_document_by_id(request: FocusedDocumentSearchRequest, context: ToolInvocationContext, *, services: DefaultToolServices | None = None):
    started = tool_started()
    tool_name = "search_document_by_id"
    services = services or get_tool_services()
    try:
        if not context.thread_id or not context.embedding_model:
            return make_tool_result(tool_name=tool_name, content="No thread context found.", context=context, started=started, warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT])
        lookup = await services.document_lookup(context.thread_id)
        if request.file_hash not in lookup:
            return make_tool_result(tool_name=tool_name, content="The requested document is not linked to this thread.", context=context, started=started, warnings=[ToolWarningCode.NO_THREAD_DOCUMENTS])
        vector = await services.embed(context.embedding_model, request.query)
        db = services.vector_db()
        raw = await db.search_knowledge_sources(thread_id=context.thread_id, query_vector=vector, embedding_model=context.embedding_model, limit=request.max_results, file_hashes=[request.file_hash], query_text=request.query)
        if not raw:
            return make_tool_result(tool_name=tool_name, content="No relevant content was found in the requested document.", context=context, started=started, warnings=[ToolWarningCode.NO_RELEVANT_CONTENT])
        if context.use_reranker:
            raw = await services.rerank(request.query, raw)
        radius = max(2, min(10, int((context.context_window or 32000) / 8000) + 1))
        ids = {i for hit in raw if hit.get("chunk_id") is not None for i in range(int(hit["chunk_id"]) - radius, int(hit["chunk_id"]) + radius + 1) if i >= 0}
        expanded = await db.get_knowledge_source_chunks_by_ids(thread_id=context.thread_id, embedding_model=context.embedding_model, file_hash=request.file_hash, chunk_ids=sorted(ids))
        expanded.sort(key=lambda item: int(item.get("chunk_id") or 0))
        from app.rag.retrieval import group_document_chunks
        content, sources = group_document_chunks(expanded, lookup)
        if not content:
            return make_tool_result(tool_name=tool_name, content="No relevant content was found in the requested document.", context=context, started=started, warnings=[ToolWarningCode.NO_RELEVANT_CONTENT])
        from app.agent.evidence_contract import evidence_segment
        return make_tool_result(tool_name=tool_name, content=content, context=context, started=started, sources=sources, artifacts={"document_sources": sources, "evidence_segments": [s for chunk in expanded if (s := evidence_segment(kind="document", content=chunk.get("text"), source=chunk, raw_score=chunk.get("rerank_score", chunk.get("score"))))]})
    except Exception as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=f"Error retrieving the requested document: {exc}")
