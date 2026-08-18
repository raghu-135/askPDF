"""Durable memory indexing and retrieval services."""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.db import (
    MemoryScopeType,
    delete_memories_for_scope,
    delete_memory,
    get_memory,
    list_memories_for_index_retry,
    list_memories,
    mark_memory_index_failed,
    mark_memory_indexed,
    mark_memory_indexing,
)
from app.db.vector import get_vector_db
from app.db.connection_sqlmodel import async_session_maker
from app.db.models_sqlmodel import GlobalMemoryRepresentation
from sqlalchemy.future import select
from app.models.llm_server_client import get_embedding_model
from app.models.memory_tools import normalize_memory_attributes
from app.models.memory_limits import MAX_MEMORY_CONTEXT_CHARS
from app.models.retry import invoke_with_retry
from app.time_utils import iso_utc_z
from app.services.embedding_model_service import (
    GLOBAL_MEMORY_EMBEDDING_MODEL,
    require_embedding_model_ready,
    resolve_thread_embedding_context,
)
from app.services.effective_memory_service import (
    memory_scope_policy_for_thread,
    resolve_effective_memory_context,
    serialize_memories_with_relationships,
)
from app.services.memory_retrieval_policy import (
    DEFAULT_RELATIVE_SCORE_RATIO,
    NEAR_DUPLICATE_TOKEN_SIMILARITY,
    memory_is_applicable,
    memory_score_floor,
    pack_memory_results,
    token_similarity,
)
from app.services.memory_repair_scheduler import schedule_global_representation_repair


logger = logging.getLogger(__name__)

_MEMORY_SCOPE_PRIORITY = {
    MemoryScopeType.THREAD.value: 0,
    MemoryScopeType.PROJECT.value: 1,
    MemoryScopeType.USER.value: 2,
}


class MemoryVectorCleanupError(RuntimeError):
    """Raised when a required memory vector cleanup operation fails."""


async def scopes_for_thread(thread_id: str, allowed_scopes: Optional[List[str]] = None) -> List[Dict[str, str]]:
    """Return policy-eligible memory scopes for a thread."""

    policy = await memory_scope_policy_for_thread(thread_id, allowed_scopes)
    return policy["searched_scopes"]


def memory_content_hash(content: str) -> str:
    return hashlib.sha256(str(content).encode("utf-8")).hexdigest()


def _merge_same_model_memory_hits(
    embedding_model: str,
    ranked_hit_groups: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Merge comparable rankings produced by one collection and query model."""

    by_id: Dict[str, Dict[str, Any]] = {}
    for hits in ranked_hit_groups:
        for hit in hits:
            memory_id = str(hit.get("memory_id") or "")
            if not memory_id:
                continue
            candidate = {
                **hit,
                "embedding_model": embedding_model,
                "raw_score": hit.get("score"),
            }
            current = by_id.get(memory_id)
            if current is None or float(candidate.get("score") or 0.0) > float(current.get("score") or 0.0):
                by_id[memory_id] = candidate
    return sorted(
        by_id.values(),
        key=lambda hit: (
            -float(hit.get("score") or 0.0),
            _MEMORY_SCOPE_PRIORITY.get(str(hit.get("scope_type") or ""), 99),
            str(hit.get("memory_id") or ""),
        ),
    )


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
            content=memory.content,
            metadata={
                "source_refs": memory.source_refs_json or {},
                "content_hash": memory.content_hash,
                "attributes": normalize_memory_attributes(memory.attributes_json),
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


async def _delete_memory_vectors(memory) -> str:
    models = [memory.embedding_model]
    if memory.scope_type == MemoryScopeType.USER.value:
        async with async_session_maker() as session:
            models.extend(list((await session.execute(
                select(GlobalMemoryRepresentation.embedding_model).where(
                    GlobalMemoryRepresentation.memory_id == memory.id
                )
            )).scalars().all()))
    for model in dict.fromkeys(models):
        deleted = await get_vector_db().delete_memory_vectors(memory.id, model)
        if not deleted:
            raise MemoryVectorCleanupError(
                f"Failed to delete memory vectors for memory {memory.id} using {model}"
            )
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
    from app.services.embedding_materialization_service import cancel_embedding_jobs_for_memory
    await cancel_embedding_jobs_for_memory(memory_id)
    deleted = await delete_memory(memory_id)
    return {"deleted": deleted, "vector_cleanup": vector_cleanup}


async def hard_delete_thread_memory_resources(thread_id: str) -> Dict[str, Any]:
    """Delete durable memory records owned by a thread."""

    # One row is sufficient to decide whether scope-wide vector cleanup is needed;
    # the repository delete below removes every canonical row in the scope.
    memories = await list_memories(
        scope_type=MemoryScopeType.THREAD.value,
        scope_id=thread_id,
        limit=1,
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


async def search_thread_memory(
    *,
    thread_id: str,
    query: str,
    allowed_scopes: Optional[List[str]] = None,
    max_results: int = 10,
    query_vector: Optional[List[float]] = None,
    char_budget: Optional[int] = None,
    score_floor: Optional[float] = None,
    relative_score_ratio: float = DEFAULT_RELATIVE_SCORE_RATIO,
) -> Dict[str, Any]:
    """Search durable memory for a thread using scope policy."""

    started = time.perf_counter()
    effective_view = await resolve_effective_memory_context(
        thread_id=thread_id,
        allowed_scopes=allowed_scopes,
    )
    policy = effective_view["policy"]
    scopes = policy["searched_scopes"]
    if not scopes:
        return {
            "memories": [],
            "scopes": [],
            "scope_policy": policy,
            "applied_overrides": [],
            "suppressed_memory_ids": [],
            "unavailable_memory_count": 0,
            "retrieval_debug": {
                "budget_chars": int(char_budget or 0),
                "candidate_limit": max_results,
                "candidate_count": 0,
                "candidates_retrieved": 0,
                "accepted_count": 0,
                "candidates_accepted": 0,
                "rejected_count": 0,
                "candidates_rejected": 0,
                "rejection_reasons": {},
                "packed_chars": 0,
                "final_packed_chars": 0,
                "searched_scopes": [],
                "skipped_scopes": policy.get("skipped_scopes", []),
                "recalled_ids": [],
                "expanded_search": False,
                "query_latency_ms": round((time.perf_counter() - started) * 1000, 2),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        }
    context = await resolve_thread_embedding_context(thread_id)
    user_scope = next((scope for scope in scopes if scope["scope_type"] == MemoryScopeType.USER.value), None)
    missing_global_ids: set[str] = set()
    if user_scope and context.embedding_model != GLOBAL_MEMORY_EMBEDDING_MODEL:
        global_ids = {
            memory.id for memory in effective_view["memory_records"]
            if memory.scope_type == MemoryScopeType.USER.value
        }
        async with async_session_maker() as session:
            indexed_global_ids = set((await session.execute(
                select(GlobalMemoryRepresentation.memory_id).where(
                    GlobalMemoryRepresentation.embedding_model == context.embedding_model,
                    GlobalMemoryRepresentation.index_status == "indexed",
                    GlobalMemoryRepresentation.memory_id.in_(global_ids),
                )
            )).scalars().all())
        missing_global_ids = global_ids - indexed_global_ids
        if missing_global_ids:
            logger.error(
                "Global memory representations are unavailable; scheduling repair | model=%s missing=%s thread_id=%s",
                context.embedding_model,
                len(missing_global_ids),
                thread_id,
            )
            schedule_global_representation_repair(context.embedding_model)

    representation_issues: List[Dict[str, Any]] = []
    if missing_global_ids:
        representation_issues.append({
            "scope_type": MemoryScopeType.USER.value,
            "embedding_model": context.embedding_model,
            "missing_count": len(missing_global_ids),
            "reason": "global_representation_warming",
        })
    await require_embedding_model_ready(context.embedding_model)
    if query_vector is None:
        embedding_client = get_embedding_model(context.embedding_model)
        query_vector = await invoke_with_retry(embedding_client.aembed_query, query)
    model_hits = await get_vector_db().search_memory(
        query_vector=query_vector,
        embedding_model=context.embedding_model,
        scope_filters=scopes,
        excluded_memory_ids=effective_view["excluded_memory_ids"],
        limit=max_results,
        query_text=query,
    )
    hits = _merge_same_model_memory_hits(context.embedding_model, [model_hits])
    effective_by_id = {memory.id: memory for memory in effective_view["memory_records"]}
    candidates: List[Dict[str, Any]] = []
    seen = set()
    rejection_reasons: Dict[str, int] = {}
    floor = memory_score_floor(context.embedding_model) if score_floor is None else max(0.0, float(score_floor))

    def reject(reason: str) -> None:
        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

    # Establish the relative cutoff from candidates that are both effective and
    # applicable. An unrelated high-scoring hit must not raise the floor for a
    # useful lower-scoring preference.
    for hit in hits:
        memory_id = hit.get("memory_id")
        if not memory_id or memory_id in seen:
            continue
        seen.add(memory_id)
        memory = effective_by_id.get(str(memory_id))
        if memory is None:
            reject("not_effective")
            continue
        score = float(hit.get("score") or 0.0)
        attributes = normalize_memory_attributes(getattr(memory, "attributes_json", None))
        if not memory_is_applicable(attributes, query):
            reject("not_applicable")
            continue
        candidates.append(
            {
                "id": memory.id,
                "scope_type": memory.scope_type,
                "scope_rank": {
                    MemoryScopeType.THREAD.value: 3,
                    MemoryScopeType.PROJECT.value: 2,
                    MemoryScopeType.USER.value: 1,
                }.get(memory.scope_type, 0),
                "scope_id": memory.scope_id,
                "content": memory.content,
                "attributes": attributes,
                "source_refs": memory.source_refs_json or {},
                "score": score,
                "score_type": hit.get("score_type") or "similarity",
                "raw_score": hit.get("raw_score"),
                "embedding_model": hit.get("embedding_model"),
                "created_at": iso_utc_z(memory.created_at) if memory.created_at else None,
                "updated_at": iso_utc_z(memory.updated_at) if memory.updated_at else None,
                "content_hash": memory.content_hash,
            }
        )

    best_score = max((float(item.get("score") or 0.0) for item in candidates), default=0.0)
    effective_floor = max(floor, best_score * max(0.0, min(1.0, relative_score_ratio)))
    relevant_candidates: List[Dict[str, Any]] = []
    for item in candidates:
        if float(item.get("score") or 0.0) < effective_floor:
            reject("below_relevance_threshold")
            continue
        relevant_candidates.append(item)

    # Deduplicate in relevance order, using narrower scope only when scores tie.
    # Final kind/recency ranking happens after the representative is selected.
    relevant_candidates.sort(key=lambda item: (
        -float(item.get("score") or 0.0),
        -int(item.get("scope_rank") or 0),
    ))
    memories: List[Dict[str, Any]] = []
    seen_hashes = set()
    for item in relevant_candidates:
        if item.get("content_hash") in seen_hashes:
            reject("exact_duplicate")
            continue
        if any(token_similarity(str(item.get("content") or ""), str(existing.get("content") or "")) >= NEAR_DUPLICATE_TOKEN_SIMILARITY for existing in memories):
            reject("near_duplicate")
            continue
        seen_hashes.add(item.get("content_hash"))
        memories.append(item)
    kind_priority = {"instruction": 0, "constraint": 1, "preference": 2, "decision": 3, "profile": 4, "fact": 5}
    def recency_rank(item: Dict[str, Any]) -> float:
        if (item.get("attributes") or {}).get("durability") != "time_sensitive":
            return 0.0
        try:
            return -datetime.fromisoformat(str(item.get("updated_at") or "").replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError):
            return 0.0

    memories.sort(key=lambda item: (
        -float(item.get("score") or 0.0),
        kind_priority.get(str((item.get("attributes") or {}).get("kind")), 9),
        -int(item.get("scope_rank") or 0),
        recency_rank(item),
    ))
    for item in memories:
        item.pop("content_hash", None)
    packed, packed_chars = pack_memory_results(memories, char_budget or MAX_MEMORY_CONTEXT_CHARS)
    rejected_count = sum(rejection_reasons.values()) + max(0, len(memories) - len(packed))
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    return {
        "memories": packed,
        "scopes": scopes,
        "scope_policy": policy,
        "applied_overrides": effective_view["applied_overrides"],
        "suppressed_memory_ids": effective_view["suppressed_memory_ids"],
        "unavailable_memory_count": effective_view["unavailable_memory_count"],
        "precedence": ["thread", "project", "user"],
        "representation_issues": representation_issues,
        "retrieval_debug": {
            "budget_chars": int(char_budget or MAX_MEMORY_CONTEXT_CHARS),
            "candidate_limit": max_results,
            "candidate_count": len(hits),
            "candidates_retrieved": len(hits),
            "accepted_count": len(packed),
            "candidates_accepted": len(packed),
            "rejected_count": rejected_count,
            "candidates_rejected": rejected_count,
            "rejection_reasons": {
                **rejection_reasons,
                **({"budget_exhausted": len(memories) - len(packed)} if len(memories) > len(packed) else {}),
            },
            "score_floor": floor,
            "effective_score_floor": effective_floor,
            "relative_score_ratio": relative_score_ratio,
            "packed_chars": packed_chars,
            "final_packed_chars": packed_chars,
            "searched_scopes": scopes,
            "skipped_scopes": policy.get("skipped_scopes", []),
            "recalled_ids": [item.get("id") for item in packed if item.get("id")],
            "expanded_search": False,
            "query_latency_ms": elapsed_ms,
            "elapsed_ms": elapsed_ms,
        },
    }


async def retry_memory_index(memory_id: str, embedding_model: str | None = None):
    """Retry one pending/failed canonical memory index operation."""

    memory = await get_memory(memory_id)
    if memory is None:
        return None
    if embedding_model and embedding_model != memory.embedding_model:
        from app.services.memory_representation_service import index_global_representation
        await index_global_representation(memory_id, embedding_model)
        return await get_memory(memory_id)
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
    return await serialize_memories_with_relationships(rows)
