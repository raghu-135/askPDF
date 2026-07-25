"""Durable memory indexing and retrieval services."""

from __future__ import annotations

from datetime import datetime
import hashlib
import logging
from typing import Any, Dict, List, Optional

from app.db import (
    MemoryScopeType,
    MemoryStatus,
    create_memory,
    delete_memories_for_scope,
    delete_memory,
    delete_memory_candidate,
    delete_memory_candidates_for_thread,
    get_memory,
    get_thread,
    list_memories_for_index_retry,
    list_expired_memories,
    list_memories,
    mark_memory_index_failed,
    mark_memory_indexed,
    mark_memory_indexing,
)
from app.db.vector import get_vector_db
from app.models.llm_server_client import get_embedding_model
from app.models.retry import invoke_with_retry
from app.time_utils import iso_utc_z
from app.services.embedding_model_service import (
    GLOBAL_MEMORY_EMBEDDING_MODEL,
    require_embedding_model_ready,
    resolve_scope_embedding_model,
    resolve_thread_embedding_context,
)


logger = logging.getLogger(__name__)


DEFAULT_MEMORY_SETTINGS = {
    "global_memory_enabled": False,
    "project_reads_user_memory": False,
    "thread_reads_project_memory": True,
    "thread_reads_user_memory": False,
}


class MemoryVectorCleanupError(RuntimeError):
    """Raised when a required memory vector cleanup operation fails."""


def _normalize_allowed_scopes(raw: Optional[List[str]]) -> set[str]:
    values = {str(item) for item in (raw or []) if item}
    if not values:
        return {MemoryScopeType.THREAD.value, MemoryScopeType.PROJECT.value}
    return values


async def scopes_for_thread(thread_id: str, allowed_scopes: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Return explicitly allowed memory scopes for a thread."""

    thread = await get_thread(thread_id)
    if thread is None or bool(getattr(thread, "is_legacy", False)):
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


def memory_content_hash(content: str) -> str:
    return hashlib.sha256(str(content).encode("utf-8")).hexdigest()


async def index_memory_record(memory) -> int:
    """Idempotently embed and index one canonical memory record."""

    await mark_memory_indexing(memory.id)
    try:
        await require_embedding_model_ready(memory.embedding_model)
        embedding_client = get_embedding_model(memory.embedding_model)
        vector = await invoke_with_retry(embedding_client.aembed_query, memory.content)
        inserted = await get_vector_db().index_memory(
            memory_id=memory.id,
            scope_type=memory.scope_type,
            scope_id=memory.scope_id,
            memory_type=memory.memory_type,
            content=memory.content,
            summary=memory.summary,
            status=memory.status,
            visibility=memory.visibility,
            metadata={
                "source_refs": memory.source_refs_json or {},
                "content_hash": memory.content_hash,
            },
            created_at=iso_utc_z(memory.created_at) if memory.created_at else None,
            updated_at=iso_utc_z(memory.updated_at) if memory.updated_at else None,
            embedding=vector,
            embedding_model=memory.embedding_model,
        )
        await mark_memory_indexed(memory.id)
        return inserted
    except Exception as exc:
        await mark_memory_index_failed(memory.id, str(exc))
        raise


async def create_and_index_memory(**kwargs):
    """Create canonical memory, then incrementally index it without losing PG state on failure."""

    embedding_model = await resolve_scope_embedding_model(
        kwargs["scope_type"], kwargs["scope_id"]
    )
    await require_embedding_model_ready(embedding_model)
    memory = await create_memory(
        embedding_model=embedding_model,
        content_hash=memory_content_hash(kwargs["content"]),
        **kwargs,
    )
    try:
        await index_memory_record(memory)
    except Exception as exc:
        logger.warning("Memory %s remains canonical with failed index state: %s", memory.id, exc)
    return await get_memory(memory.id)


async def _delete_memory_vectors(memory) -> str:
    deleted = await get_vector_db().delete_memory_vectors(memory.id, memory.embedding_model)
    if not deleted:
        raise MemoryVectorCleanupError(f"Failed to delete memory vectors for memory {memory.id}")
    return "deleted"


async def _delete_scope_memory_vectors(
    *,
    scope_type: str,
    scope_id: str,
    embedding_model: str,
) -> str:
    deleted = await get_vector_db().delete_memory_vectors_for_scope(scope_type, scope_id, embedding_model)
    if not deleted:
        raise MemoryVectorCleanupError(f"Failed to delete memory vectors for scope {scope_type}:{scope_id}")
    return "deleted"


async def hard_delete_memory(memory_id: str) -> Dict[str, Any]:
    """Hard-delete one memory after required vector cleanup, when the vector target is known."""

    memory = await get_memory(memory_id)
    if memory is None:
        return {"deleted": False, "vector_cleanup": "not_found"}
    vector_cleanup = await _delete_memory_vectors(memory)
    deleted = await delete_memory(memory_id)
    return {"deleted": deleted, "vector_cleanup": vector_cleanup}


async def hard_delete_memory_candidate(candidate_id: str) -> Dict[str, Any]:
    """Hard-delete one memory promotion candidate."""

    return {"deleted": await delete_memory_candidate(candidate_id)}


async def hard_delete_thread_memory_resources(thread_id: str) -> Dict[str, Any]:
    """Delete durable memory records and candidates owned by a thread."""

    memories = await list_memories(
        scope_type=MemoryScopeType.THREAD.value,
        scope_id=thread_id,
        status="",
        limit=500,
    )
    vector_cleanup = "skipped"
    if memories:
        context = await resolve_thread_embedding_context(thread_id)
        vector_cleanup = await _delete_scope_memory_vectors(
            scope_type=MemoryScopeType.THREAD.value,
            scope_id=thread_id,
            embedding_model=context.embedding_model,
        )
    deleted_memory_ids = await delete_memories_for_scope(
        scope_type=MemoryScopeType.THREAD.value,
        scope_id=thread_id,
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
    limit: int = 100,
) -> Dict[str, Any]:
    """Hard-delete expired memories after required vector cleanup, when vector targets are known."""

    expired = await list_expired_memories(now=now, limit=limit)
    vector_cleanup = {"deleted": 0, "skipped": 0}
    for memory in expired:
        cleanup_result = await _delete_memory_vectors(memory)
        vector_cleanup[cleanup_result] = vector_cleanup.get(cleanup_result, 0) + 1
    deleted_ids = []
    for memory in expired:
        if await delete_memory(memory.id):
            deleted_ids.append(memory.id)
    return {
        "deleted_memory_ids": deleted_ids,
        "vector_cleanup": vector_cleanup,
    }


async def search_thread_memory(
    *,
    thread_id: str,
    query: str,
    allowed_scopes: Optional[List[str]] = None,
    max_results: int = 10,
) -> Dict[str, Any]:
    """Search durable memory for a thread using scope policy."""

    scopes = await scopes_for_thread(thread_id, allowed_scopes)
    if not scopes:
        return {"memories": [], "scopes": []}
    context = await resolve_thread_embedding_context(thread_id)
    if not context.long_term_memory_enabled:
        return {"memories": [], "scopes": []}
    scopes_by_model: Dict[str, List[Dict[str, str]]] = {}
    for scope in scopes:
        model = (
            GLOBAL_MEMORY_EMBEDDING_MODEL
            if scope["scope_type"] == MemoryScopeType.USER.value
            else context.embedding_model
        )
        scopes_by_model.setdefault(model, []).append(scope)

    hits = []
    for embedding_model, model_scopes in scopes_by_model.items():
        await require_embedding_model_ready(embedding_model)
        embedding_client = get_embedding_model(embedding_model)
        query_vector = await invoke_with_retry(embedding_client.aembed_query, query)
        model_hits = await get_vector_db().search_memory(
            query_vector=query_vector,
            embedding_model=embedding_model,
            scope_filters=model_scopes,
            limit=max_results,
            query_text=query,
        )
        hits.extend(model_hits)
    hits.sort(key=lambda item: float(item.get("score") or 0), reverse=True)
    memories = []
    seen = set()
    for hit in hits:
        memory_id = hit.get("memory_id")
        if not memory_id or memory_id in seen:
            continue
        seen.add(memory_id)
        memory = await get_memory(str(memory_id))
        if (
            memory is None
            or memory.status != MemoryStatus.ACTIVE.value
            or (memory.expires_at is not None and memory.expires_at <= datetime.now(memory.expires_at.tzinfo))
        ):
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
        if len(memories) >= max_results:
            break
    return {"memories": memories, "scopes": scopes}


async def retry_memory_index(memory_id: str):
    """Retry one pending/failed canonical memory index operation."""

    memory = await get_memory(memory_id)
    if memory is None:
        return None
    if memory.index_status == "indexed":
        return memory
    await index_memory_record(memory)
    return await get_memory(memory_id)


async def retry_pending_memory_indexes(*, limit: int = 100) -> Dict[str, Any]:
    """Incrementally retry only pending/failed rows; this is not a full recalculation."""

    rows = await list_memories_for_index_retry(limit=limit)
    indexed_ids: List[str] = []
    failed_ids: List[str] = []
    for memory in rows:
        try:
            await index_memory_record(memory)
            indexed_ids.append(memory.id)
        except Exception:
            failed_ids.append(memory.id)
    return {"indexed_ids": indexed_ids, "failed_ids": failed_ids}


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
            "embedding_model": memory.embedding_model,
            "content_hash": memory.content_hash,
            "index_status": memory.index_status,
            "index_attempts": memory.index_attempts,
            "indexed_at": iso_utc_z(memory.indexed_at) if memory.indexed_at else None,
            "index_error": memory.index_error,
            "visibility": memory.visibility,
            "created_by": memory.created_by,
            "fork_origin": memory.fork_origin_json,
            "created_at": iso_utc_z(memory.created_at) if memory.created_at else None,
            "updated_at": iso_utc_z(memory.updated_at) if memory.updated_at else None,
        }
        for memory in rows
    ]
