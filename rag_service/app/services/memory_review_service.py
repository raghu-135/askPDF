"""Memory consistency activity, review status, and similarity candidate groups."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Sequence

from sqlalchemy import and_, or_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import MemoryScopeType
from app.db.models_sqlmodel import GlobalMemoryRepresentation, Memory, MemoryOverride, MemoryReviewState, MemoryScopeActivity, Project, Thread
from app.db.vector import get_vector_db
from app.models.llm_server_client import get_embedding_model
from app.models.retry import invoke_with_retry
from app.services.embedding_model_service import GLOBAL_MEMORY_EMBEDDING_MODEL, require_embedding_model_ready
from app.services.memory_policy import LOCAL_USER_MEMORY_SCOPE_ID
from app.time_utils import iso_utc_z, utc_now


SCOPE_RANK = {"thread": 3, "project": 2, "user": 1}


def scope_key(scope_type: str, scope_id: str) -> str:
    return f"{scope_type}:{scope_id}"


async def bump_memory_scope_activity(
    scopes: Sequence[tuple[str, str]],
    *,
    session: AsyncSession,
) -> None:
    now = utc_now()
    for scope_type, scope_id in sorted(set(scopes)):
        stmt = pg_insert(MemoryScopeActivity).values(
            scope_type=scope_type,
            scope_id=scope_id,
            version=1,
            changed_at=now,
        ).on_conflict_do_update(
            index_elements=["scope_type", "scope_id"],
            set_={
                "version": MemoryScopeActivity.version + 1,
                "changed_at": now,
            },
        )
        await session.execute(stmt)


async def _review_context(context_type: str, context_id: str):
    async with async_session_maker() as session:
        if context_type == "thread":
            thread = await session.get(Thread, context_id)
            if thread is None:
                raise ValueError("Thread not found")
            project = await session.get(Project, thread.project_id)
            scopes = [
                ("thread", thread.id),
                ("project", thread.project_id),
                ("user", LOCAL_USER_MEMORY_SCOPE_ID),
            ]
            return scopes, thread.embedding_model
        if context_type == "project":
            project = await session.get(Project, context_id)
            if project is None:
                raise ValueError("Project not found")
            return [("project", project.id), ("user", LOCAL_USER_MEMORY_SCOPE_ID)], project.embedding_model
    raise ValueError("Memory review requires a project or thread context")


async def get_memory_review_status(context_type: str, context_id: str) -> Dict[str, Any]:
    scopes, embedding_model = await _review_context(context_type, context_id)
    async with async_session_maker() as session:
        activity_rows = list((await session.execute(select(MemoryScopeActivity).where(or_(*[
            and_(MemoryScopeActivity.scope_type == scope_type, MemoryScopeActivity.scope_id == scope_id)
            for scope_type, scope_id in scopes
        ])))).scalars().all())
        state = await session.get(MemoryReviewState, (context_type, context_id))
    current = {scope_key(row.scope_type, row.scope_id): row.version for row in activity_rows}
    reviewed = dict(state.reviewed_scope_versions_json or {}) if state else {}
    status = "never_reviewed" if state is None else (
        "review_suggested" if any(int(current.get(key, 0)) > int(reviewed.get(key, 0)) for key in current) else "current"
    )
    return {
        "context_type": context_type,
        "context_id": context_id,
        "status": status,
        "embedding_model": embedding_model,
        "current_scope_versions": current,
        "reviewed_scope_versions": reviewed,
        "last_reviewed_at": iso_utc_z(state.last_reviewed_at) if state and state.last_reviewed_at else None,
    }


async def complete_memory_review(context_type: str, context_id: str, versions: Dict[str, int]) -> None:
    now = utc_now()
    async with async_session_maker() as session:
        async with session.begin():
            stmt = pg_insert(MemoryReviewState).values(
                context_type=context_type,
                context_id=context_id,
                reviewed_scope_versions_json=versions,
                last_reviewed_at=now,
                created_at=now,
                updated_at=now,
            ).on_conflict_do_update(
                index_elements=["context_type", "context_id"],
                set_={
                    "reviewed_scope_versions_json": versions,
                    "last_reviewed_at": now,
                    "updated_at": now,
                },
            )
            await session.execute(stmt)


async def build_memory_review_batch(
    context_type: str,
    context_id: str,
    *,
    anchor_position: int = 0,
    max_clusters: int = 5,
    snapshot_at: datetime | None = None,
    snapshot_scope_versions: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    """Build bounded similarity groups from stable, scope-aware anchors."""

    scopes, model = await _review_context(context_type, context_id)
    status = await get_memory_review_status(context_type, context_id)
    cutoff = snapshot_at or utc_now()
    clauses = [and_(Memory.scope_type == kind, Memory.scope_id == ident) for kind, ident in scopes]
    async with async_session_maker() as session:
        review_state = await session.get(MemoryReviewState, (context_type, context_id))
        all_memories = list((await session.execute(
            select(Memory).where(or_(*clauses))
            .order_by(Memory.scope_type, Memory.updated_at.asc().nullsfirst(), Memory.created_at, Memory.id)
        )).scalars().all())
        candidate_global_ids = {
            memory.id for memory in all_memories
            if memory.scope_type == MemoryScopeType.USER.value
        }
        represented_global_ids = (
            set((await session.execute(
                select(GlobalMemoryRepresentation.memory_id).where(
                    GlobalMemoryRepresentation.memory_id.in_(candidate_global_ids),
                    GlobalMemoryRepresentation.embedding_model == model,
                    GlobalMemoryRepresentation.index_status == "indexed",
                )
            )).scalars().all())
            if candidate_global_ids and model != GLOBAL_MEMORY_EMBEDDING_MODEL
            else {
                memory.id for memory in all_memories
                if memory.scope_type == MemoryScopeType.USER.value and memory.index_status == "indexed"
            }
        )
        all_memories = [
            memory for memory in all_memories
            if memory.index_status == "indexed" or memory.id in represented_global_ids
        ]
        anchors = [
            memory for memory in all_memories
            if not review_state or not review_state.last_reviewed_at
            or (memory.updated_at or memory.created_at) > review_state.last_reviewed_at
            if (memory.updated_at or memory.created_at) <= cutoff
        ]
        ids = [memory.id for memory in all_memories]
        edges = list((await session.execute(select(MemoryOverride).where(or_(
            MemoryOverride.overriding_memory_id.in_(ids), MemoryOverride.overridden_memory_id.in_(ids)
        )))).scalars().all()) if ids else []
        global_ids = {memory.id for memory in all_memories if memory.scope_type == MemoryScopeType.USER.value}
    by_id = {memory.id: memory for memory in all_memories}
    start = max(0, anchor_position)
    selected_anchors = anchors[start:start + max_clusters]
    degraded = represented_global_ids != global_ids
    groups: List[Dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    try:
        await require_embedding_model_ready(model)
        embedder = get_embedding_model(model)
        scope_filters = [{"scope_type": kind, "scope_id": ident} for kind, ident in scopes]
        for anchor in selected_anchors:
            vector = await invoke_with_retry(embedder.aembed_query, anchor.content)
            hits = await get_vector_db().search_memory(
                query_vector=vector,
                embedding_model=model,
                scope_filters=scope_filters,
                excluded_memory_ids=[anchor.id],
                limit=7,
                query_text=anchor.content,
            )
            if degraded and global_ids:
                await require_embedding_model_ready(GLOBAL_MEMORY_EMBEDDING_MODEL)
                fallback_vector = await invoke_with_retry(
                    get_embedding_model(GLOBAL_MEMORY_EMBEDDING_MODEL).aembed_query,
                    anchor.content,
                )
                hits.extend(await get_vector_db().search_memory(
                    query_vector=fallback_vector,
                    embedding_model=GLOBAL_MEMORY_EMBEDDING_MODEL,
                    scope_filters=[{"scope_type": "user", "scope_id": LOCAL_USER_MEMORY_SCOPE_ID}],
                    excluded_memory_ids=[anchor.id],
                    limit=7,
                    query_text=anchor.content,
                ))
            related_ids = [str(hit.get("memory_id")) for hit in hits if str(hit.get("memory_id")) in by_id]
            related_ids.extend(
                edge.overridden_memory_id if edge.overriding_memory_id == anchor.id else edge.overriding_memory_id
                for edge in edges
                if anchor.id in {edge.overriding_memory_id, edge.overridden_memory_id}
            )
            member_ids = [anchor.id]
            for related_id in related_ids:
                pair = tuple(sorted((anchor.id, related_id)))
                if related_id != anchor.id and pair not in seen_pairs:
                    seen_pairs.add(pair)
                    member_ids.append(related_id)
                if len(member_ids) >= 8:
                    break
            if len(member_ids) > 1:
                candidate = {
                    "anchor_id": anchor.id,
                    "memories": [{
                        "id": by_id[item].id,
                        "scope_type": by_id[item].scope_type,
                        "scope_id": by_id[item].scope_id,
                        "scope_rank": SCOPE_RANK[by_id[item].scope_type],
                        "content": by_id[item].content,
                        "updated_at": iso_utc_z(by_id[item].updated_at or by_id[item].created_at),
                    } for item in member_ids],
                }
                candidate_ids = {item["id"] for item in candidate["memories"]}
                overlapping = next((
                    group for group in groups
                    if candidate_ids.intersection(item["id"] for item in group["memories"])
                ), None)
                if overlapping is None:
                    groups.append(candidate)
                else:
                    existing_ids = {item["id"] for item in overlapping["memories"]}
                    overlapping["memories"].extend(
                        item for item in candidate["memories"]
                        if item["id"] not in existing_ids
                    )
                    overlapping["memories"] = overlapping["memories"][:8]
    except Exception:
        degraded = True
    next_position = start + len(selected_anchors)
    return {
        "context_type": context_type,
        "context_id": context_id,
        "snapshot_at": iso_utc_z(cutoff),
        "snapshot_scope_versions": snapshot_scope_versions or status["current_scope_versions"],
        "anchor_position": next_position,
        "reviewed_anchor_count": next_position,
        "remaining_anchor_count": max(0, len(anchors) - next_position),
        "candidate_groups": groups,
        "degraded": degraded,
        "embedding_model": model,
    }
