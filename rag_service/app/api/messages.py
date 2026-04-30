"""
Messages API Module - Message and chat endpoints.

Endpoints:
- GET /api/threads/{thread_id}/messages - List messages
- DELETE /api/messages/{message_id} - Delete message
- POST /api/threads/{thread_id}/chat - Thread chat
"""

import traceback
from typing import Optional

from fastapi import APIRouter, HTTPException

from app.agent.agent import normalize_tool_instructions
from app.db import (
    MessageRole,
    delete_message_pair,
    get_message,
    get_thread,
    get_thread_messages,
    get_thread_settings,
    recompute_qa_stats,
)
from app.db.vector import get_vector_db
from app.models.llm_server_client import (
    INTENT_AGENT_MAX_ITERATIONS,
    merge_thread_settings,
)
from app.models.requests import ThreadChatRequest
from app.rag.chat_service import handle_thread_chat

router = APIRouter(tags=["messages"])


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages_endpoint(
    thread_id: str, limit: int = 100, offset: int = 0
):
    """Get messages for a thread with pagination."""
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        messages = await get_thread_messages(thread_id, limit, offset)
        return {
            "thread_id": thread_id,
            "messages": [
                {
                    "id": m.id,
                    "role": m.role.value if hasattr(m.role, 'value') else str(m.role),
                    "content": m.content,
                    "context_compact": m.context_compact,
                    "reasoning": m.reasoning,
                    "reasoning_available": m.reasoning_available,
                    "reasoning_format": m.reasoning_format,
                    "web_sources": m.web_sources,
                    "created_at": m.created_at.isoformat(),
                }
                for m in messages
            ],
            "limit": limit,
            "offset": offset,
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/messages/{message_id}")
async def delete_message_endpoint(message_id: str):
    """
    Delete a message and its associated chat memory from Weaviate.
    If it's part of a QA pair, deletes both messages, their chat-memory vector,
    and any web search chunks (web_search type) whose URLs are no longer referenced
    by any other message in the thread.
    """
    try:
        # Get message to find thread_id and role
        message = await get_message(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        # Identify both sides of the QA pair
        all_msgs = await get_thread_messages(message.thread_id, limit=10000)
        assistant_msg_id = None
        if message.role == MessageRole.ASSISTANT:
            assistant_msg_id = message_id
        else:
            # USER → find the immediately following assistant message
            for i, m in enumerate(all_msgs):
                if (
                    m.id == message_id
                    and i + 1 < len(all_msgs)
                    and all_msgs[i + 1].role == MessageRole.ASSISTANT
                ):
                    assistant_msg_id = all_msgs[i + 1].id
                    break

        # IDs that will be removed from SQLite (this + its pair counterpart)
        ids_to_delete: set = {message_id}
        if assistant_msg_id and assistant_msg_id != message_id:
            ids_to_delete.add(assistant_msg_id)

        # Collect web_source URLs from the assistant message being deleted
        urls_to_check: set = set()
        if assistant_msg_id:
            asst_msg = await get_message(assistant_msg_id)
            if asst_msg and asst_msg.web_sources:
                for ws in asst_msg.web_sources:
                    url = ws.get("url", "").strip()
                    if url:
                        urls_to_check.add(url)

        db = get_vector_db()

        # Delete chat-memory vector
        vector_message_id = assistant_msg_id or message_id
        await db.delete_chat_memory_by_message_id(message.thread_id, vector_message_id)

        # Delete orphaned web_search chunks
        if urls_to_check:
            # URLs still referenced by other (surviving) messages
            still_needed: set = set()
            for m in all_msgs:
                if m.id not in ids_to_delete and m.web_sources:
                    for ws in m.web_sources:
                        url = ws.get("url", "").strip()
                        if url:
                            still_needed.add(url)
            orphaned = urls_to_check - still_needed
            if orphaned:
                await db.delete_web_chunks_by_urls(message.thread_id, list(orphaned))

        # Delete from SQLite (pair-aware)
        deleted_ids = await delete_message_pair(message_id)

        # Recompute QA stats to reflect the deletion
        try:
            await recompute_qa_stats(message.thread_id)
        except Exception as stats_err:
            import logging
            logging.getLogger(__name__).warning(f"thread_stats recompute skipped after delete: {stats_err}")

        return {"status": "deleted", "deleted_ids": deleted_ids}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threads/{thread_id}/chat")
async def thread_chat_endpoint(thread_id: str, req: ThreadChatRequest):
    """
    Thread-based chat with semantic memory.
    Returns answer, used_chat_ids (recollected messages), and document_sources.
    """
    try:
        # Verify thread exists
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        # Override thread_id from path
        req.thread_id = thread_id
        thread_settings = merge_thread_settings(await get_thread_settings(thread_id))
        if req.max_iterations is None:
            req.max_iterations = thread_settings["max_iterations"]
        if req.system_role_override is None:
            req.system_role_override = thread_settings["system_role"]
        if req.tool_instructions_override is None:
            req.tool_instructions_override = normalize_tool_instructions(
                thread_settings.get("tool_instructions", {})
            )
        if req.custom_instructions_override is None:
            req.custom_instructions_override = thread_settings["custom_instructions"]
        if req.use_intent_agent is None:
            req.use_intent_agent = thread_settings.get("use_intent_agent", True)
        if req.intent_agent_max_iterations is None:
            req.intent_agent_max_iterations = thread_settings.get(
                "intent_agent_max_iterations", INTENT_AGENT_MAX_ITERATIONS
            )
        if req.reasoning_mode is None:
            req.reasoning_mode = thread_settings.get("reasoning_mode", True)

        result = await handle_thread_chat(thread_id, req, thread.embed_model)
        return result
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
