"""Framework-neutral conversation-history retrieval."""

from app.agent.tool_contract import ToolWarningCode, make_tool_error_result, make_tool_result, tool_started
from app.tools.contracts import DocumentSearchRequest
from app.tools.context import ToolInvocationContext
from app.tools.services import DefaultToolServices, get_tool_services


async def search_thread_conversation_history(request: DocumentSearchRequest, context: ToolInvocationContext, *, services: DefaultToolServices | None = None):
    started = tool_started()
    tool_name = "search_thread_conversation_history"
    services = services or get_tool_services()
    try:
        if not context.thread_id or not context.embedding_model:
            return make_tool_result(tool_name=tool_name, content="No thread context found.", context=context, started=started, warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT])
        vector = await services.embed(context.embedding_model, request.query)
        result = await services.semantic_history(thread_id=context.thread_id, query_vector=vector, query_text=request.query, limit=request.max_results, use_reranker=context.use_reranker, embedding_model=context.embedding_model, include_refs=True)
        history, used_ids = result[:2]
        refs = result[2] if len(result) > 2 else []
        if not history:
            return make_tool_result(tool_name=tool_name, content="No relevant past conversations found.", context=context, started=started, warnings=[ToolWarningCode.NO_RELEVANT_CONVERSATION_HISTORY])
        from app.agent.evidence_contract import evidence_segment
        return make_tool_result(tool_name=tool_name, content=history, context=context, started=started, artifacts={"used_chat_ids": used_ids, "evidence_segments": [s for item in refs if (s := evidence_segment(kind="conversation", content=item.get("content"), source=item, raw_score=item.get("rerank_score", item.get("score"))))]})
    except Exception as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=f"Error retrieving chat memory: {exc}")
