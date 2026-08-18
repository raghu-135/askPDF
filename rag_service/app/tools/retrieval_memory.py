"""Framework-neutral durable-memory retrieval."""

from typing import Any

from app.agent.tool_contract import ToolWarningCode, make_tool_error_result, make_tool_result, tool_started
from app.tools.contracts import DocumentSearchRequest
from app.tools.context import ToolInvocationContext


async def search_durable_memory(request: DocumentSearchRequest, context: ToolInvocationContext, *, services: Any = None):
    del services
    started = tool_started()
    tool_name = "search_durable_memory"
    try:
        if not context.thread_id:
            return make_tool_result(tool_name=tool_name, content="No thread context found.", context=context, started=started, warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT])
        from app.models.memory_tools import MEMORY_READ_EFFECTIVE, MAX_MEMORY_CONTEXT_CHARS, MemorySearchInput
        from app.services.memory_tool_service import build_memory_tool_context, search_memory_tool
        memory_context, _thread, _project = await build_memory_tool_context(selected_scope_type="thread", selected_scope_id=context.thread_id, thread_id=context.thread_id, capabilities=[MEMORY_READ_EFFECTIVE])
        prefetched = context.prefetched_durable_memories
        policy = context.prefetched_durable_memory_scope_policy or {}
        debug = context.prefetched_durable_memory_debug or {}
        reuse = isinstance(prefetched, list) and bool(policy.get("searched_scopes")) and len(prefetched) >= 3 and not int((debug.get("rejection_reasons") or {}).get("budget_exhausted") or 0)
        if reuse:
            result = {"memories": prefetched, "scopes": context.prefetched_durable_memory_scopes or [], "scope_policy": policy, "retrieval_debug": {**debug, "expanded_search": False, "reused_prefetch": True}}
        else:
            char_budget = min(
                int((context.context_window or 32000) * 4),
                MAX_MEMORY_CONTEXT_CHARS,
            )
            result = await search_memory_tool(
                memory_context,
                MemorySearchInput(
                    query=request.query,
                    view="effective",
                    max_results=request.max_results,
                    char_budget=char_budget,
                ),
                query_vector=context.prefetched_durable_memory_query_vector,
            )
        memories = result.get("memories", []) if isinstance(result, dict) else []
        scopes = result.get("scopes", []) if isinstance(result, dict) else []
        policy = result.get("scope_policy", {}) if isinstance(result, dict) else {}
        if not memories:
            return make_tool_result(tool_name=tool_name, content="No relevant long-term memories found.", context=context, started=started, warnings=[ToolWarningCode.NO_RELEVANT_MEMORY], artifacts={"memory_refs": [], "memory_scopes": scopes, "memory_scope_policy": policy})
        lines = ["[LONG-TERM MEMORY]", "These memories are defaults. Explicit instructions in the current user request override them for this run. Otherwise Thread overrides Project, and Project overrides Personal memory."]
        refs, segments = [], []
        from app.agent.evidence_contract import evidence_segment
        for index, item in enumerate(memories, 1):
            memory = item if isinstance(item, dict) else {}
            scope_type, scope_id = memory.get("scope_type") or "unknown", memory.get("scope_id") or "unknown"
            content = memory.get("excerpt") or memory.get("content") or ""
            lines.append(f"{index}. {scope_type}:{scope_id} ({(memory.get('attributes') or {}).get('kind', 'fact')}): {content}")
            refs.append({
                "memory_id": memory.get("id"),
                "scope_type": scope_type,
                "scope_id": scope_id,
                "score": memory.get("score"),
                "score_type": memory.get("score_type"),
                "raw_score": memory.get("raw_score"),
                "embedding_model": memory.get("embedding_model"),
                "scope_rank": memory.get("scope_rank"),
                "attributes": memory.get("attributes") or {},
            })
            if segment := evidence_segment(kind="memory", content=content, source={"memory_id": memory.get("id"), "title": f"{scope_type}:{scope_id}"}, raw_score=memory.get("raw_score", memory.get("score"))):
                segments.append(segment)
        return make_tool_result(tool_name=tool_name, content="\n".join(lines), context=context, started=started, artifacts={"memory_refs": refs, "memory_scopes": scopes, "memory_scope_policy": policy, "memory_retrieval_debug": result.get("retrieval_debug", {}), "evidence_segments": segments})
    except Exception as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=f"Error retrieving long-term memory: {exc}")
