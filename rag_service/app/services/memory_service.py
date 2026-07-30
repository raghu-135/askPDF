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
    get_memory,
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
    EmbeddingModelResolutionError,
    GLOBAL_MEMORY_EMBEDDING_MODEL,
    require_embedding_model_ready,
    resolve_scope_embedding_model,
    resolve_thread_embedding_context,
)
from app.services.memory_policy import (
    LOCAL_USER_MEMORY_SCOPE_ID,
    normalize_project_memory_settings,
    normalize_thread_memory_settings,
)


logger = logging.getLogger(__name__)

MEMORY_RRF_K = 60
_MEMORY_SCOPE_PRIORITY = {
    MemoryScopeType.THREAD.value: 0,
    MemoryScopeType.PROJECT.value: 1,
    MemoryScopeType.USER.value: 2,
}


class MemoryVectorCleanupError(RuntimeError):
    """Raised when a required memory vector cleanup operation fails."""


def _normalize_allowed_scopes(raw: Optional[List[str]]) -> set[str]:
    if raw is None:
        return {
            MemoryScopeType.THREAD.value,
            MemoryScopeType.PROJECT.value,
            MemoryScopeType.USER.value,
        }
    return {str(item) for item in raw if item}


async def memory_scope_policy_for_thread(
    thread_id: str,
    allowed_scopes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Resolve requested scopes against project and thread consent gates."""

    try:
        context = await resolve_thread_embedding_context(thread_id)
    except EmbeddingModelResolutionError:
        return {"requested_scopes": [], "searched_scopes": [], "skipped_scopes": []}

    thread = context.thread
    project = context.project
    thread_settings = normalize_thread_memory_settings(thread.settings)
    project_settings = normalize_project_memory_settings(project.settings_json)
    allowed = _normalize_allowed_scopes(allowed_scopes)
    scopes: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []
    ordered_scope_types = [
        MemoryScopeType.THREAD.value,
        MemoryScopeType.PROJECT.value,
        MemoryScopeType.USER.value,
    ]
    requested = [scope_type for scope_type in ordered_scope_types if scope_type in allowed]

    if MemoryScopeType.THREAD.value not in allowed:
        skipped.append({"scope_type": MemoryScopeType.THREAD.value, "reason": "not_requested"})
    else:
        scopes.append({"scope_type": MemoryScopeType.THREAD.value, "scope_id": thread.id})

    if MemoryScopeType.PROJECT.value not in allowed:
        skipped.append({"scope_type": MemoryScopeType.PROJECT.value, "reason": "not_requested"})
    elif not thread_settings["thread_reads_project_memory"]:
        skipped.append({"scope_type": MemoryScopeType.PROJECT.value, "reason": "thread_opt_out"})
    else:
        scopes.append({"scope_type": MemoryScopeType.PROJECT.value, "scope_id": project.id})

    if MemoryScopeType.USER.value not in allowed:
        skipped.append({"scope_type": MemoryScopeType.USER.value, "reason": "not_requested"})
    elif not project_settings["project_reads_user_memory"]:
        skipped.append({"scope_type": MemoryScopeType.USER.value, "reason": "project_opt_out"})
    elif not thread_settings["thread_reads_user_memory"]:
        skipped.append({"scope_type": MemoryScopeType.USER.value, "reason": "thread_opt_out"})
    else:
        scopes.append({
            "scope_type": MemoryScopeType.USER.value,
            "scope_id": LOCAL_USER_MEMORY_SCOPE_ID,
        })

    return {
        "requested_scopes": requested,
        "searched_scopes": scopes,
        "skipped_scopes": skipped,
    }


async def scopes_for_thread(thread_id: str, allowed_scopes: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Return policy-eligible memory scopes for a thread."""

    policy = await memory_scope_policy_for_thread(thread_id, allowed_scopes)
    return policy["searched_scopes"]


def memory_content_hash(content: str) -> str:
    return hashlib.sha256(str(content).encode("utf-8")).hexdigest()


def _rank_fuse_memory_hits(
    ranked_hit_groups: List[tuple[str, List[Dict[str, Any]]]],
) -> List[Dict[str, Any]]:
    """Fuse model-local rankings without comparing their raw similarity scores."""

    fused: Dict[str, Dict[str, Any]] = {}
    for model_order, (embedding_model, hits) in enumerate(ranked_hit_groups):
        for rank, hit in enumerate(hits, start=1):
            memory_id = str(hit.get("memory_id") or "")
            if not memory_id:
                continue
            contribution = 1.0 / (MEMORY_RRF_K + rank)
            existing = fused.get(memory_id)
            if existing is None:
                existing = {
                    **hit,
                    "embedding_model": embedding_model,
                    "raw_score": hit.get("score"),
                    "score": 0.0,
                    "_best_rank": rank,
                    "_model_order": model_order,
                }
                fused[memory_id] = existing
            existing["score"] += contribution
            existing["_best_rank"] = min(existing["_best_rank"], rank)

    ranked = sorted(
        fused.values(),
        key=lambda hit: (
            -float(hit["score"]),
            _MEMORY_SCOPE_PRIORITY.get(str(hit.get("scope_type") or ""), 99),
            int(hit["_best_rank"]),
            int(hit["_model_order"]),
            str(hit.get("memory_id") or ""),
        ),
    )
    for hit in ranked:
        hit.pop("_best_rank", None)
        hit.pop("_model_order", None)
    return ranked


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

    if kwargs["scope_type"] == MemoryScopeType.USER.value:
        kwargs["scope_id"] = LOCAL_USER_MEMORY_SCOPE_ID
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


async def hard_delete_thread_memory_resources(thread_id: str) -> Dict[str, Any]:
    """Delete durable memory records owned by a thread."""

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
    return {
        "deleted_memory_ids": deleted_memory_ids,
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

    policy = await memory_scope_policy_for_thread(thread_id, allowed_scopes)
    scopes = policy["searched_scopes"]
    if not scopes:
        return {"memories": [], "scopes": [], "scope_policy": policy}
    context = await resolve_thread_embedding_context(thread_id)
    scopes_by_model: Dict[str, List[Dict[str, str]]] = {}
    for scope in scopes:
        model = (
            GLOBAL_MEMORY_EMBEDDING_MODEL
            if scope["scope_type"] == MemoryScopeType.USER.value
            else context.embedding_model
        )
        scopes_by_model.setdefault(model, []).append(scope)

    ranked_hit_groups: List[tuple[str, List[Dict[str, Any]]]] = []
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
        ranked_hit_groups.append((embedding_model, model_hits))
    hits = _rank_fuse_memory_hits(ranked_hit_groups)
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
                "score_type": "rrf",
                "raw_score": hit.get("raw_score"),
                "embedding_model": hit.get("embedding_model"),
                "created_at": iso_utc_z(memory.created_at) if memory.created_at else None,
                "updated_at": iso_utc_z(memory.updated_at) if memory.updated_at else None,
            }
        )
        if len(memories) >= max_results:
            break
    return {"memories": memories, "scopes": scopes, "scope_policy": policy}


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
