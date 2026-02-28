"""
main.py - FastAPI entrypoint for the RAG Service

This module provides endpoints for:
- Thread management (create, list, get, delete)
- Document indexing (per-thread and legacy)
- Chat with retrieval-augmented generation (with semantic memory)
- Message management (list, delete)
- Model listing and health checks

Dependencies:
- FastAPI
- httpx
- dotenv
- rag, agent, vectordb.qdrant, database (local modules)
"""

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage

from agent import app as agent_app, get_tool_catalog, normalize_tool_instructions, build_system_prompt
from rag import index_document, index_document_for_thread
from vectordb.qdrant import QdrantAdapter
from models import (
    check_chat_model_ready,
    check_embed_model_ready,
    fetch_available_models,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOKEN_BUDGET,
    MIN_MAX_ITERATIONS,
    MAX_MAX_ITERATIONS,
    MAX_CUSTOM_INSTRUCTIONS_CHARS,
    MAX_SYSTEM_ROLE_CHARS,
    merge_thread_settings,
)
from chat_service import handle_chat, handle_thread_chat
from database import (
    init_db, 
    create_thread, get_thread, list_threads, update_thread, delete_thread,
    get_thread_settings, update_thread_settings,
    create_or_get_file, get_file, add_file_to_thread, get_thread_files, is_file_in_thread,
    create_message, get_message, get_thread_messages, delete_message, delete_message_pair, get_recent_messages,
    MessageRole
)

# Load environment variables from .env file
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    await init_db()
    yield


app = FastAPI(title="RAG Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Request/Response Models ============

class IndexRequest(BaseModel):
    """Request body for /index endpoint."""
    embedding_model: str
    metadata: Optional[Dict[str, Any]] = None


class ThreadCreateRequest(BaseModel):
    """Request body for creating a thread."""
    name: str
    embed_model: str


class ThreadUpdateRequest(BaseModel):
    """Request body for updating a thread."""
    name: str


class ThreadFileRequest(BaseModel):
    """Request body for adding a file to a thread."""
    file_hash: str
    file_name: str
    file_path: Optional[str] = None


class ChatRequest(BaseModel):
    """Request body for /chat endpoint (legacy)."""
    question: str
    llm_model: str
    embedding_model: str
    collection_name: Optional[str] = None
    use_web_search: bool = False
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    history: List[Dict[str, str]] = []  # list of {role: "user"|"assistant", content: "..."}


class ThreadChatRequest(BaseModel):
    """Request body for thread-based chat."""
    thread_id: str
    question: str
    llm_model: str
    use_web_search: bool = False
    context_window: int = DEFAULT_TOKEN_BUDGET  # Added context window size
    max_iterations: Optional[int] = Field(default=None, ge=MIN_MAX_ITERATIONS, le=MAX_MAX_ITERATIONS)
    system_role_override: Optional[str] = Field(default=None, max_length=MAX_SYSTEM_ROLE_CHARS)
    tool_instructions_override: Optional[Dict[str, str]] = None
    custom_instructions_override: Optional[str] = Field(default=None, max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS)


class ThreadSettingsResponse(BaseModel):
    max_iterations: int = Field(default=DEFAULT_MAX_ITERATIONS, ge=MIN_MAX_ITERATIONS, le=MAX_MAX_ITERATIONS)
    system_role: str = Field(default="", max_length=MAX_SYSTEM_ROLE_CHARS)
    tool_instructions: Dict[str, str] = Field(default_factory=dict)
    custom_instructions: str = Field(default="", max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS)


class ThreadSettingsUpdateRequest(BaseModel):
    max_iterations: Optional[int] = Field(default=None, ge=MIN_MAX_ITERATIONS, le=MAX_MAX_ITERATIONS)
    system_role: Optional[str] = Field(default=None, max_length=MAX_SYSTEM_ROLE_CHARS)
    tool_instructions: Optional[Dict[str, str]] = None
    custom_instructions: Optional[str] = Field(default=None, max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS)


class ToolCatalogEntry(BaseModel):
    id: str
    display_name: str
    description: str
    default_prompt: str


class PromptDefaults(BaseModel):
    max_iterations: int
    system_role: str
    tool_instructions: Dict[str, str]
    custom_instructions: str


class PromptPreviewRequest(BaseModel):
    context_window: int = DEFAULT_TOKEN_BUDGET
    system_role: Optional[str] = Field(default=None, max_length=MAX_SYSTEM_ROLE_CHARS)
    tool_instructions: Optional[Dict[str, str]] = None
    custom_instructions: Optional[str] = Field(default=None, max_length=MAX_CUSTOM_INSTRUCTIONS_CHARS)


# ============ Thread Endpoints ============

@app.get("/prompt-tools")
async def prompt_tools_endpoint():
    """Return user-facing tool aliases and default prompts for prompt customization UI."""
    try:
        defaults = merge_thread_settings({})
        defaults["tool_instructions"] = normalize_tool_instructions(defaults.get("tool_instructions", {}))
        payload = PromptDefaults(**defaults).dict()
        return {
            "tools": [ToolCatalogEntry(**t).dict() for t in get_tool_catalog()],
            "defaults": payload,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prompt-preview")
async def prompt_preview_endpoint(req: PromptPreviewRequest):
    """Return the fully composed system prompt preview from the backend source of truth."""
    try:
        tool_instructions = normalize_tool_instructions(req.tool_instructions or {})
        prompt = build_system_prompt(
            context_window=req.context_window,
            system_role=req.system_role or "",
            tool_instructions=tool_instructions,
            custom_instructions=req.custom_instructions or "",
        )
        return {"prompt": prompt}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/threads")
async def create_thread_endpoint(req: ThreadCreateRequest):
    """
    Create a new chat thread.
    Also creates the Qdrant collection for this thread.
    """
    try:
        # Create thread in SQLite
        thread = await create_thread(req.name, req.embed_model)
        
        # Create Qdrant collection for the thread
        db = QdrantAdapter()
        # Default vector size will be determined when first embeddings are added
        # For now, we'll create it on first index
        
        return {
            "id": thread.id,
            "name": thread.name,
            "embed_model": thread.embed_model,
            "settings": thread.settings,
            "created_at": thread.created_at.isoformat()
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads")
async def list_threads_endpoint():
    """
    List all threads with message and file counts.
    """
    try:
        threads = await list_threads()
        return {"threads": threads}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}")
async def get_thread_endpoint(thread_id: str):
    """
    Get a specific thread by ID.
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        files = await get_thread_files(thread_id)
        db = QdrantAdapter()
        stats = await db.get_thread_stats(thread_id)
        
        return {
            "id": thread.id,
            "name": thread.name,
            "embed_model": thread.embed_model,
            "settings": thread.settings,
            "created_at": thread.created_at.isoformat(),
            "files": [{"file_hash": f.file_hash, "file_name": f.file_name} for f in files],
            "stats": stats,
            "file_count": len(files)
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/threads/{thread_id}")
async def update_thread_endpoint(thread_id: str, req: ThreadUpdateRequest):
    """
    Update a thread's name.
    Note: embed_model cannot be changed once set.
    """
    try:
        thread = await update_thread(thread_id, req.name)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        return {
            "id": thread.id,
            "name": thread.name,
            "embed_model": thread.embed_model,
            "settings": thread.settings,
            "created_at": thread.created_at.isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}/settings")
async def get_thread_settings_endpoint(thread_id: str):
    """Get persisted chat settings for a thread."""
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        settings = merge_thread_settings(await get_thread_settings(thread_id))
        settings["tool_instructions"] = normalize_tool_instructions(settings.get("tool_instructions", {}))
        return ThreadSettingsResponse(**settings)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/threads/{thread_id}/settings")
async def update_thread_settings_endpoint(thread_id: str, req: ThreadSettingsUpdateRequest):
    """Update persisted chat settings for a thread."""
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        current = merge_thread_settings(await get_thread_settings(thread_id))
        updates = req.dict(exclude_none=True)
        next_settings = {**current, **updates}
        next_settings["tool_instructions"] = normalize_tool_instructions(next_settings.get("tool_instructions", {}))
        persisted = await update_thread_settings(thread_id, next_settings)
        if persisted is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        merged = merge_thread_settings(persisted)
        merged["tool_instructions"] = normalize_tool_instructions(merged.get("tool_instructions", {}))
        return ThreadSettingsResponse(**merged)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/threads/{thread_id}")
async def delete_thread_endpoint(thread_id: str):
    """
    Delete a thread, its messages, file associations, and Qdrant collection.
    """
    try:
        # Delete Qdrant collection first
        db = QdrantAdapter()
        await db.delete_thread_collection(thread_id)
        
        # Delete from SQLite
        deleted = await delete_thread(thread_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        return {"status": "deleted", "thread_id": thread_id}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ Thread File Endpoints ============

@app.post("/threads/{thread_id}/files")
async def add_file_to_thread_endpoint(
    thread_id: str, 
    req: ThreadFileRequest,
    background_tasks: BackgroundTasks
):
    """
    Add a file to a thread and trigger background indexing.
    """
    try:
        # Verify thread exists
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        # Create or get file record
        file = await create_or_get_file(req.file_hash, req.file_name, req.file_path)
        
        # Associate file with thread
        await add_file_to_thread(thread_id, req.file_hash)
        
        # Check if already indexed in this thread's collection
        db = QdrantAdapter()
        collection_exists = await db.thread_collection_exists(thread_id)
        
        # Trigger background indexing
        background_tasks.add_task(
            index_document_for_thread,
            thread_id=thread_id,
            file_hash=req.file_hash,
            embedding_model_name=thread.embed_model
        )
        
        return {
            "status": "accepted",
            "thread_id": thread_id,
            "file_hash": req.file_hash,
            "file_name": req.file_name,
            "indexing": "in_progress"
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}/files")
async def get_thread_files_endpoint(thread_id: str):
    """
    Get all files associated with a thread.
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        files = await get_thread_files(thread_id)
        return {
            "thread_id": thread_id,
            "files": [{"file_hash": f.file_hash, "file_name": f.file_name, "file_path": f.file_path} for f in files]
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ Message Endpoints ============

@app.get("/threads/{thread_id}/messages")
async def get_thread_messages_endpoint(
    thread_id: str,
    limit: int = 100,
    offset: int = 0
):
    """
    Get messages for a thread with pagination.
    """
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
                    "role": m.role.value,
                    "content": m.content,
                    "reasoning": m.reasoning,
                    "reasoning_available": m.reasoning_available,
                    "reasoning_format": m.reasoning_format,
                    "created_at": m.created_at.isoformat()
                }
                for m in messages
            ],
            "limit": limit,
            "offset": offset
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/messages/{message_id}")
async def delete_message_endpoint(message_id: str):
    """
    Delete a message and its associated chat memory from Qdrant.
    If it's part of a QA pair, deletes both messages, their chat-memory vector,
    and any web search chunks (web_search type) whose URLs are no longer referenced
    by any other message in the thread.
    """
    try:
        # Get message to find thread_id and role
        message = await get_message(message_id)
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")

        # ── Identify both sides of the QA pair ──
        all_msgs = await get_thread_messages(message.thread_id, limit=10000)
        assistant_msg_id = None
        if message.role == MessageRole.ASSISTANT:
            assistant_msg_id = message_id
        else:
            # USER → find the immediately following assistant message
            for i, m in enumerate(all_msgs):
                if m.id == message_id and i + 1 < len(all_msgs) and all_msgs[i + 1].role == MessageRole.ASSISTANT:
                    assistant_msg_id = all_msgs[i + 1].id
                    break

        # IDs that will be removed from SQLite (this + its pair counterpart)
        ids_to_delete: set = {message_id}
        if assistant_msg_id and assistant_msg_id != message_id:
            ids_to_delete.add(assistant_msg_id)

        # ── Collect web_source URLs from the assistant message being deleted ──
        urls_to_check: set = set()
        if assistant_msg_id:
            asst_msg = await get_message(assistant_msg_id)
            if asst_msg and asst_msg.web_sources:
                for ws in asst_msg.web_sources:
                    url = ws.get("url", "").strip()
                    if url:
                        urls_to_check.add(url)

        db = QdrantAdapter()

        # ── Delete chat-memory vector ──
        qdrant_msg_id = assistant_msg_id or message_id
        await db.delete_chat_memory_by_message_id(message.thread_id, qdrant_msg_id)

        # ── Delete orphaned web_search chunks ──
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

        # ── Delete from SQLite (pair-aware) ──
        deleted_ids = await delete_message_pair(message_id)

        return {"status": "deleted", "deleted_ids": deleted_ids}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ Chat Endpoints ============

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Legacy chat endpoint for retrieval-augmented generation.
    Use /threads/{thread_id}/chat for thread-based chat.
    """
    try:
        result = await handle_chat(req)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/threads/{thread_id}/chat")
async def thread_chat_endpoint(thread_id: str, req: ThreadChatRequest):
    """
    Thread-based chat with semantic memory.
    Returns answer, used_chat_ids (recollected messages), and pdf_sources.
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
            req.tool_instructions_override = normalize_tool_instructions(thread_settings.get("tool_instructions", {}))
        if req.custom_instructions_override is None:
            req.custom_instructions_override = thread_settings["custom_instructions"]
        
        result = await handle_thread_chat(thread_id, req, thread.embed_model)
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============ Indexing Endpoints ============

@app.post("/index")
async def index_endpoint(req: IndexRequest):
    """
    Legacy index endpoint for document indexing.
    For thread-based indexing, use POST /threads/{thread_id}/files.
    """
    try:
        result = await index_document(
            embedding_model_name=req.embedding_model,
            metadata=req.metadata,
        )
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}/index-status")
async def get_thread_index_status(thread_id: str, file_hash: Optional[str] = None):
    """
    Check indexing status for a thread (or specific file in thread).
    """
    try:
        thread = await get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")
        
        db = QdrantAdapter()
        
        if file_hash:
            # Check specific file
            is_indexed = await db.has_file_indexed(thread_id, file_hash)
            status = "ready" if is_indexed else "not_ready"
        else:
            # Check all files in thread
            files = await get_thread_files(thread_id)
            if not files:
                status = "ready"
            else:
                all_indexed = True
                for f in files:
                    # Check if file chunks exist for this thread
                    if not await db.has_file_indexed(thread_id, f.file_hash):
                        all_indexed = False
                        break
                status = "ready" if all_indexed else "not_ready"
        
        stats = await db.get_thread_stats(thread_id)
        
        return {
            "thread_id": thread_id,
            "status": status,
            "stats": stats
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def status_endpoint(collection_name: str):
    """
    Legacy: Check if a collection exists and is ready in the vector database.
    """
    try:
        db = QdrantAdapter()
        exists = await db.collection_exists(collection_name)
        return {"status": "ready" if exists else "not_ready", "collection": collection_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============ Model Endpoints ============

@app.get("/models")
async def get_models():
    """
    Fetch available LLM and embedding models from the LLM API/server.
    """
    return await fetch_available_models()


@app.get("/health/is_chat_model_ready")
async def is_chat_model_ready_endpoint(model: str):
    """
    Check if a chat/LLM model is ready.
    """
    ready = await check_chat_model_ready(model)
    return {"model": model, "chat_model_ready": ready}


@app.get("/health/is_embed_model_ready")
async def is_embed_model_ready_endpoint(model: str):
    """
    Check if an embedding model is ready.
    """
    ready = await check_embed_model_ready(model)
    return {"model": model, "embed_model_ready": ready}


@app.get("/health")
async def health():
    """
    Service health check endpoint.
    """
    return {"status": "ok"}
