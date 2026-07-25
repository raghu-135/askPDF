"""Durable memory indexing and retrieval services."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.db import (
    MemoryScopeType,
    MemoryStatus,
    create_memory,
    get_memory,
    get_thread,
    list_memories,
)
from app.db.vector import get_vector_db
from app.models.llm_server_client import get_embedding_model
from app.models.retry import invoke_with_retry
from app.time_utils import iso_utc_z


DEFAULT_MEMORY_SETTINGS = {
    "global_memory_enabled": False,
    "project_reads_user_memory": False,
    "thread_reads_project_memory": True,
    "thread_reads_user_memory": False,
}


def _normalize_allowed_scopes(raw: Optional[List[str]]) -> set[str]:
    values = {str(item) for item in (raw or []) if item}
    if not values:
        return {MemoryScopeType.THREAD.value, MemoryScopeType.PROJECT.value}
    return values


async def scopes_for_thread(thread_id: str, allowed_scopes: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Return explicitly allowed memory scopes for a thread."""

    thread = await get_thread(thread_id)
    if thread is None:
        return []
    settings = {**DEFAULT_MEMORY_SETTINGS, **(thread.settings or {}).get("memory", {})}
    allowed = _normalize_allowed_scopes(allowed_scopes)
    scopes: List[Dict[str, str]] = []
    if MemoryScopeType.THREAD.value in allowed:
        scopes.append({"scope_type": MemoryScopeType.THREAD.value, "scope_id": thread.id})
    if (
        MemoryScopeType.PROJECT.value in allowed
        and thread.project_id
        and settings.get("thread_reads_project_memory", True)
    ):
        scopes.append({"scope_type": MemoryScopeType.PROJECT.value, "scope_id": thread.project_id})
    if (
        MemoryScopeType.USER.value in allowed
        and settings.get("global_memory_enabled")
        and settings.get("thread_reads_user_memory")
    ):
        user_id = (thread.thread_metadata or {}).get("user_id") or "default"
        scopes.append({"scope_type": MemoryScopeType.USER.value, "scope_id": str(user_id)})
    return scopes


async def index_memory_record(memory, embedding_model: str) -> int:
    """Embed and index one durable memory record."""

    embedding_client = get_embedding_model(embedding_model)
    vector = await invoke_with_retry(embedding_client.aembed_query, memory.content)
    return await get_vector_db().index_memory(
        memory_id=memory.id,
        scope_type=memory.scope_type,
        scope_id=memory.scope_id,
        memory_type=memory.memory_type,
        content=memory.content,
        summary=memory.summary,
        status=memory.status,
        visibility=memory.visibility,
        metadata={"source_refs": memory.source_refs_json or {}},
        created_at=iso_utc_z(memory.created_at) if memory.created_at else None,
        updated_at=iso_utc_z(memory.updated_at) if memory.updated_at else None,
        embedding=vector,
        embedding_model=embedding_model,
    )


async def create_and_index_memory(*, embedding_model: Optional[str] = None, **kwargs):
    """Create a memory and index it when an embedding model is available."""

    memory = await create_memory(**kwargs)
    if embedding_model:
        await index_memory_record(memory, embedding_model)
    return memory


async def search_thread_memory(
    *,
    thread_id: str,
    query: str,
    embedding_model: str,
    allowed_scopes: Optional[List[str]] = None,
    max_results: int = 10,
) -> Dict[str, Any]:
    """Search durable memory for a thread using scope policy."""

    scopes = await scopes_for_thread(thread_id, allowed_scopes)
    if not scopes:
        return {"memories": [], "scopes": []}
    embedding_client = get_embedding_model(embedding_model)
    query_vector = await invoke_with_retry(embedding_client.aembed_query, query)
    hits = await get_vector_db().search_memory(
        query_vector=query_vector,
        embedding_model=embedding_model,
        scope_filters=scopes,
        limit=max_results,
        query_text=query,
    )
    memories = []
    seen = set()
    for hit in hits:
        memory_id = hit.get("memory_id")
        if not memory_id or memory_id in seen:
            continue
        seen.add(memory_id)
        memory = await get_memory(str(memory_id))
        if memory is None or memory.status != MemoryStatus.ACTIVE.value:
            continue
        memories.append(
            {
                "id": memory.id,
                "scope_type": memory.scope_type,
                "scope_id": memory.scope_id,
                "memory_type": memory.memory_type,
                "content": memory.content,
                "summary": memory.summary,
                "source_refs": memory.source_refs_json or {},
                "confidence": memory.confidence,
                "visibility": memory.visibility,
                "score": hit.get("score"),
                "created_at": iso_utc_z(memory.created_at) if memory.created_at else None,
                "updated_at": iso_utc_z(memory.updated_at) if memory.updated_at else None,
            }
        )
    return {"memories": memories, "scopes": scopes}


async def list_scope_memories(*, scope_type: str, scope_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    rows = await list_memories(scope_type=scope_type, scope_id=scope_id, limit=limit)
    return [
        {
            "id": memory.id,
            "scope_type": memory.scope_type,
            "scope_id": memory.scope_id,
            "memory_type": memory.memory_type,
            "content": memory.content,
            "summary": memory.summary,
            "source_refs": memory.source_refs_json or {},
            "confidence": memory.confidence,
            "status": memory.status,
            "visibility": memory.visibility,
            "created_by": memory.created_by,
            "fork_origin": memory.fork_origin_json,
            "created_at": iso_utc_z(memory.created_at) if memory.created_at else None,
            "updated_at": iso_utc_z(memory.updated_at) if memory.updated_at else None,
        }
        for memory in rows
    ]
