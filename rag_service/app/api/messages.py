"""
Messages API Module - Message and chat endpoints.

Endpoints:
- GET /api/threads/{thread_id}/messages - List messages
- DELETE /api/messages/{message_id} - Delete message
- POST /api/threads/{thread_id}/chat - Thread chat
"""

import asyncio
import json
import traceback
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse

from app.agent.prompting import normalize_tool_instructions
from app.agent_workflows.repository import AgentWorkflowRepository
from app.agent_workflows.service import AgentRunService
from app.agent_workflows.execution_stream import AgentExecutionEventSink, retain_background_task
from app.agent_workflows.workflow_runtime import workflow_supports_replans
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
from app.time_utils import iso_utc_z
from app.models.llm_server_client import merge_thread_settings
from app.models.requests import ThreadChatRequest
from app.services.embedding_model_service import (
    EmbeddingModelResolutionError,
    EmbeddingModelUnavailableError,
    require_thread_embedding_ready,
)

router = APIRouter(tags=["messages"])


async def _settings_workflow_supports_replans(settings: dict) -> bool:
    agent_workflow = settings.get("agent_workflow")
    workflow_id = agent_workflow.get("workflow_id") if isinstance(agent_workflow, dict) else None
    if not isinstance(workflow_id, str) or not workflow_id:
        return False
    workflow = await AgentWorkflowRepository().get_workflow(workflow_id, include_custom=True)
    spec = workflow.spec_json if workflow and isinstance(workflow.spec_json, dict) else {}
    return workflow_supports_replans(spec)


def _agent_message_metadata(message) -> dict:
    metadata = getattr(message, "metadata", None)
    if not isinstance(metadata, dict):
        return {}
    allowed_keys = {
        "agent_workflow_id",
        "agent_route",
        "agent_route_reason",
    }
    return {key: metadata[key] for key in allowed_keys if key in metadata}


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
                    "metadata": _agent_message_metadata(m),
                    "agent_run_id": getattr(m, "agent_run_id", None),
                    "agent_run_turn_kind": getattr(m, "agent_run_turn_kind", None),
                    "agent_run_sequence": getattr(m, "agent_run_sequence", None),
                    "agent_trace_refs": getattr(m, "agent_trace_refs", None),
                    "created_at": iso_utc_z(m.created_at),
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
            return {"status": "not_found", "deleted_ids": []}

        # Identify both sides of the QA pair
        all_msgs = await get_thread_messages(message.thread_id, limit=10000)
        assistant_msg_id = None
        if message.role == MessageRole.ASSISTANT.value:
            assistant_msg_id = message_id
        else:
            # USER → find the immediately following assistant message
            for i, m in enumerate(all_msgs):
                if (
                    m.id == message_id
                    and i + 1 < len(all_msgs)
                    and all_msgs[i + 1].role == MessageRole.ASSISTANT.value
                ):
                    assistant_msg_id = all_msgs[i + 1].id
                    break

        # IDs that will be removed from database (this + its pair counterpart)
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

        # Get thread to determine embedding model
        thread = await get_thread(message.thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        db = get_vector_db()

        # Delete chat-memory vector
        vector_message_id = getattr(message, "turn_id", None) or (
            (assistant_msg_id or message_id).split(":")[0]
        )
        await db.delete_chat_memory_by_message_id(message.thread_id, vector_message_id, thread.embedding_model)

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
                await db.delete_web_chunks_by_urls(message.thread_id, list(orphaned), thread.embedding_model)

        # Delete from database (pair-aware)
        deleted_ids = await delete_message_pair(message_id)

        # Recompute QA stats to reflect the deletion
        try:
            await recompute_qa_stats(message.thread_id)
        except Exception as stats_err:
            import logging
            logging.getLogger(__name__).warning(f"thread stats recompute skipped after delete: {stats_err}")

        return {"status": "deleted", "deleted_ids": deleted_ids}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/threads/{thread_id}/chat")
async def thread_chat_endpoint(
    thread_id: str,
    req: ThreadChatRequest,
    accept: Optional[str] = Header(default=None),
):
    """
    Thread-based chat with semantic memory.
    Returns answer, used_chat_ids (recollected messages), and document_sources.
    """
    try:
        try:
            embedding_context = await require_thread_embedding_ready(thread_id)
        except EmbeddingModelResolutionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except EmbeddingModelUnavailableError as exc:
            raise HTTPException(
                status_code=409,
                detail={"code": "embedding_model_unavailable", "message": str(exc)},
            ) from exc
        thread = embedding_context.thread

        # Override thread_id from path
        req.thread_id = thread_id
        thread_settings = merge_thread_settings(await get_thread_settings(thread_id))
        if req.replans is None and await _settings_workflow_supports_replans(thread_settings):
            req.replans = thread_settings["replans"]
        if req.system_role_override is None:
            req.system_role_override = thread_settings["system_role"]
        if req.tool_instructions_override is None:
            req.tool_instructions_override = normalize_tool_instructions(
                thread_settings.get("tool_instructions", {})
            )
        if req.custom_instructions_override is None:
            req.custom_instructions_override = thread_settings["custom_instructions"]
        service = AgentRunService()
        if "text/event-stream" not in str(accept or "").lower():
            return await service.run_thread_chat(thread_id, req, embedding_context.embedding_model)

        sink = AgentExecutionEventSink(include_details=False)

        async def run_chat() -> None:
            try:
                result = await service.run_thread_chat(
                    thread_id,
                    req,
                    embedding_context.embedding_model,
                    execution_event_sink=sink,
                )
                await sink.queue.put({"event": "__result__", "data": result})
            except Exception as exc:
                traceback.print_exc()
                await sink.queue.put({
                    "event": "__error__",
                    "data": {"error": {"code": "chat_stream_failed", "raw_message": str(exc), "retryable": True}},
                })

        async def events():
            sequence = 0
            task = asyncio.create_task(run_chat())
            retain_background_task(task)
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(sink.queue.get(), timeout=12)
                    except asyncio.TimeoutError:
                        sequence += 1
                        yield _chat_sse({"event": "heartbeat", "data": {}}, sequence)
                        continue
                    event = str(item.get("event") or "message")
                    data = item.get("data") or {}
                    if event == "__result__":
                        break
                    if event == "__error__":
                        sequence += 1
                        yield _chat_sse({"event": "stream.error", "data": data}, sequence)
                        break
                    sequence += 1
                    yield _chat_sse(item, sequence)
                    if event in {"run.completed", "run.failed", "run.cancelled"}:
                        break
            finally:
                sink.detach_delivery()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _chat_sse(event: dict[str, Any], sequence: int) -> str:
    name = str(event.get("event") or "message")
    payload = {"id": sequence, "event": name, "data": event.get("data") or {}}
    return f"id: {sequence}\nevent: {name}\ndata: {json.dumps(payload, default=str)}\n\n"
