"""Durable memory indexing and retrieval services."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.db import (
    MemoryScopeType,
    MemoryStatus,
    create_memory,
    delete_expired_memories,
    delete_memories_for_scope,
    delete_memory,
    delete_memory_candidate,
    delete_memory_candidates_for_thread,
    get_memory,
    get_thread,
    list_memories,
)
from app.db.vector import get_vector_db
from app.models.llm_server_client import get_embedding_model
from app.models.retry import invoke_with_retry
from app.time_utils import iso_utc_z


logger = logging.getLogger(__name__)


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


async def _embedding_model_for_memory(memory, embedding_model: Optional[str] = None) -> Optional[str]:
    if embedding_model:
        return embedding_model
    if memory.scope_type != MemoryScopeType.THREAD.value:
        return None
    thread = await get_thread(memory.scope_id)
    return thread.embedding_model if thread is not None else None


async def _best_effort_delete_memory_vectors(memory, embedding_model: Optional[str]) -> bool:
    model = await _embedding_model_for_memory(memory, embedding_model)
    if not model:
        return False
    try:
        await get_vector_db().delete_memory_vectors(memory.id, model)
        return True
    except Exception:
        logger.exception("Failed to delete memory vectors for memory %s", memory.id)
        return False


async def _best_effort_delete_scope_memory_vectors(
    *,
    scope_type: str,
    scope_id: str,
    embedding_model: Optional[str],
) -> bool:
    if not embedding_model:
        return False
    try:
        await get_vector_db().delete_memory_vectors_for_scope(scope_type, scope_id, embedding_model)
        return True
    except Exception:
        logger.exception("Failed to delete memory vectors for scope %s:%s", scope_type, scope_id)
        return False


async def hard_delete_memory(memory_id: str, *, embedding_model: Optional[str] = None) -> Dict[str, Any]:
    """Hard-delete one memory and best-effort remove its vector row."""

    memory = await get_memory(memory_id)
    if memory is None:
        return {"deleted": False, "vector_cleanup": False}
    vector_cleanup = await _best_effort_delete_memory_vectors(memory, embedding_model)
    deleted = await delete_memory(memory_id)
    return {"deleted": deleted, "vector_cleanup": vector_cleanup}


async def hard_delete_memory_candidate(candidate_id: str) -> Dict[str, Any]:
    """Hard-delete one memory promotion candidate."""

    return {"deleted": await delete_memory_candidate(candidate_id)}


async def hard_delete_thread_memory_resources(thread_id: str, *, embedding_model: str) -> Dict[str, Any]:
    """Delete durable memory records and candidates owned by a thread."""

    deleted_memory_ids = await delete_memories_for_scope(
        scope_type=MemoryScopeType.THREAD.value,
        scope_id=thread_id,
    )
    vector_cleanup = False
    if deleted_memory_ids:
        vector_cleanup = await _best_effort_delete_scope_memory_vectors(
            scope_type=MemoryScopeType.THREAD.value,
            scope_id=thread_id,
            embedding_model=embedding_model,
        )
    deleted_candidate_ids = await delete_memory_candidates_for_thread(thread_id)
    return {
        "deleted_memory_ids": deleted_memory_ids,
        "deleted_candidate_ids": deleted_candidate_ids,
        "vector_cleanup": vector_cleanup,
    }


async def hard_delete_expired_memories(
    *,
    now: Optional[datetime] = None,
    embedding_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Hard-delete expired memories and best-effort remove their vector rows."""

    expired = await delete_expired_memories(now=now)
    cleaned_vectors = 0
    for memory in expired:
        if await _best_effort_delete_memory_vectors(memory, embedding_model):
            cleaned_vectors += 1
    return {
        "deleted_memory_ids": [memory.id for memory in expired],
        "vector_cleanup_count": cleaned_vectors,
    }


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
