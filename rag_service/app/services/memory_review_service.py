"""Memory consistency activity, review status, and similarity candidate groups."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Sequence

from sqlalchemy import and_, func, or_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import ChatTurnStatus, MemoryScopeType
from app.db.models_sqlmodel import ChatTurn, GlobalMemoryRepresentation, Memory, MemoryOverride, MemoryReviewState, MemoryScopeActivity, Project, Thread
from app.db.vector import get_vector_db
from app.models.llm_server_client import get_embedding_model
from app.models.memory_manager_input_budget import (
    MAX_MEMORY_SEARCH_QUERY_CHARS,
    MAX_REVIEW_FETCH_ROWS,
    compute_memory_manager_input_budget,
)
from app.models.retry import invoke_with_retry
from app.services.embedding_model_service import GLOBAL_MEMORY_EMBEDDING_MODEL, require_embedding_model_ready
from app.services.memory_policy import LOCAL_USER_MEMORY_SCOPE_ID
from app.services.memory_repair_scheduler import schedule_global_representation_repair
from app.time_utils import iso_utc_z, utc_now


SCOPE_RANK = {"thread": 3, "project": 2, "user": 1}
logger = logging.getLogger(__name__)
MAX_REVIEW_ANCHORS = 100
REVIEW_ANCHOR_ESTIMATED_CHARS = 1500
MAX_REVIEW_NEIGHBORS = 7
MAX_REVIEW_GROUP_MEMORIES = 8


def _turn_text(turn: ChatTurn) -> Dict[str, Any]:
    payload = turn.payload if isinstance(turn.payload, dict) else {}
    return {
        "id": turn.id,
        "question": str(payload.get("question") or ""),
        "answer": str(payload.get("answer") or ""),
        "created_at": iso_utc_z(turn.created_at),
    }


async def build_conversation_review_batch(
    thread: Thread | None,
    *,
    context_window: int,
    session_factory=async_session_maker,
) -> Dict[str, Any] | None:
    """Return the next stable batch of completed turns for memory curation."""

    if thread is None:
        return None
    metadata = thread.thread_metadata if isinstance(thread.thread_metadata, dict) else {}
    curator_meta = metadata.get("memory_curator") if isinstance(metadata.get("memory_curator"), dict) else {}
    cursor_at_raw = curator_meta.get("reviewed_through_created_at")
    cursor_id = str(curator_meta.get("reviewed_through_turn_id") or "")
    cursor_at = None
    if cursor_at_raw:
        try:
            cursor_at = datetime.fromisoformat(str(cursor_at_raw).replace("Z", "+00:00"))
        except ValueError:
            cursor_at = None

    base_filters = [
        ChatTurn.thread_id == thread.id,
        ChatTurn.status == ChatTurnStatus.COMPLETED.value,
        ChatTurn.payload["question"].astext.isnot(None),
        ChatTurn.payload["question"].astext != "",
        ChatTurn.payload["answer"].astext.isnot(None),
        ChatTurn.payload["answer"].astext != "",
    ]
    review_budget = compute_memory_manager_input_budget(context_window)["review_context_chars"]
    async with session_factory() as session:
        if cursor_at is None:
            result = await session.execute(
                select(ChatTurn)
                .where(*base_filters)
                .order_by(ChatTurn.created_at.desc(), ChatTurn.id.desc())
                .limit(MAX_REVIEW_FETCH_ROWS)
            )
            candidates = list(result.scalars().all())
            remaining = 0
        else:
            after_cursor = or_(
                ChatTurn.created_at > cursor_at,
                and_(ChatTurn.created_at == cursor_at, ChatTurn.id > cursor_id),
            )
            result = await session.execute(
                select(ChatTurn)
                .where(*base_filters, after_cursor)
                .order_by(ChatTurn.created_at.asc(), ChatTurn.id.asc())
                .limit(MAX_REVIEW_FETCH_ROWS)
            )
            candidates = list(result.scalars().all())

        turns = []
        used_chars = 0
        for candidate in candidates:
            serialized = json.dumps(_turn_text(candidate), ensure_ascii=True)
            if turns and used_chars + len(serialized) > review_budget:
                break
            turns.append(candidate)
            used_chars += len(serialized)

        if cursor_at is None:
            turns.reverse()
        else:
            if turns:
                last = turns[-1]
                remaining = int((await session.execute(
                    select(func.count(ChatTurn.id)).where(
                        *base_filters,
                        or_(
                            ChatTurn.created_at > last.created_at,
                            and_(ChatTurn.created_at == last.created_at, ChatTurn.id > last.id),
                        ),
                    )
                )).scalar() or 0)
            else:
                remaining = 0
    reviewed_through = turns[-1] if turns else None
    return {
        "turns": [_turn_text(turn) for turn in turns],
        "reviewed_count": len(turns),
        "remaining_count": remaining,
        "context_budget_chars": review_budget,
        "cursor": (
            {
                "thread_id": thread.id,
                "reviewed_through_turn_id": reviewed_through.id,
                "reviewed_through_created_at": iso_utc_z(reviewed_through.created_at),
            }
            if reviewed_through else None
        ),
    }


def build_review_memory_search_query(
    messages: Sequence[Dict[str, Any]],
    conversation_review: Dict[str, Any] | None = None,
    *,
    max_chars: int = MAX_MEMORY_SEARCH_QUERY_CHARS,
) -> str:
    """Build a bounded similarity query while retaining evidence from every reviewed turn."""

    message_text = "\n".join(
        str(item.get("content") or "").strip()
        for item in messages[-4:]
        if str(item.get("content") or "").strip()
    )[: min(max_chars, 2000)]
    turns = list((conversation_review or {}).get("turns") or [])
    if not turns:
        return message_text[:max_chars]

    separator_budget = len(turns) - 1 + (1 if message_text else 0)
    remaining = max(0, max_chars - len(message_text) - separator_budget)
    per_turn = max(1, remaining // len(turns)) if turns else 0
    turn_text = []
    for turn in turns:
        text = (
            f'Question: {str(turn.get("question") or "").strip()}\n'
            f'Answer: {str(turn.get("answer") or "").strip()}'
        ).strip()
        turn_text.append(text[:per_turn])
    return "\n".join(([message_text] if message_text else []) + turn_text)[:max_chars]


def _visible_review_neighbors(anchor_id: str, hit_ids, edges, visible_by_id) -> List[str]:
    """Return deterministic review neighbors limited to the resolved context."""

    candidates = [str(memory_id) for memory_id in hit_ids]
    candidates.extend(
        edge.overridden_memory_id if edge.overriding_memory_id == anchor_id else edge.overriding_memory_id
        for edge in edges
        if anchor_id in {edge.overriding_memory_id, edge.overridden_memory_id}
    )
    visible = []
    seen = set()
    for memory_id in candidates:
        if memory_id == anchor_id or memory_id not in visible_by_id or memory_id in seen:
            continue
        seen.add(memory_id)
        visible.append(memory_id)
    return visible


def _review_override_edges(memory_ids, edges) -> List[Dict[str, str]]:
    member_ids = set(memory_ids)
    return [
        {
            "overriding_memory_id": edge.overriding_memory_id,
            "overridden_memory_id": edge.overridden_memory_id,
        }
        for edge in edges
        if edge.overriding_memory_id in member_ids and edge.overridden_memory_id in member_ids
    ]


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
        if context_type == "user":
            if context_id != LOCAL_USER_MEMORY_SCOPE_ID:
                raise ValueError("Global memory review context not found")
            return [("user", LOCAL_USER_MEMORY_SCOPE_ID)], GLOBAL_MEMORY_EMBEDDING_MODEL
    raise ValueError("Memory review requires a user, project, or thread context")


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
    context_window: int,
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
    review_budget = compute_memory_manager_input_budget(context_window)["review_context_chars"]
    max_anchors = min(
        MAX_REVIEW_ANCHORS,
        max(1, review_budget // REVIEW_ANCHOR_ESTIMATED_CHARS),
    )
    selected_anchors = anchors[start:start + max_anchors]
    missing_global_ids = global_ids - represented_global_ids
    representation_pending = bool(missing_global_ids)
    if representation_pending and model != GLOBAL_MEMORY_EMBEDDING_MODEL:
        logger.error(
            "Global memory representations are unavailable for consistency review; scheduling repair | "
            "context=%s:%s model=%s missing=%s",
            context_type,
            context_id,
            model,
            len(missing_global_ids),
        )
        schedule_global_representation_repair(model)
        return {
            "context_type": context_type,
            "context_id": context_id,
            "snapshot_at": iso_utc_z(cutoff),
            "snapshot_scope_versions": snapshot_scope_versions or status["current_scope_versions"],
            "anchor_position": start,
            "reviewed_anchor_count": start,
            "remaining_anchor_count": max(0, len(anchors) - start),
            "candidate_groups": [],
            "representation_pending": True,
            "missing_representation_count": len(missing_global_ids),
            "embedding_model": model,
            "blocked": True,
        }
    groups: List[Dict[str, Any]] = []
    group_chars = 0
    processed_anchor_count = 0
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
                limit=MAX_REVIEW_NEIGHBORS,
                query_text=anchor.content,
            )
            related_ids = _visible_review_neighbors(
                anchor.id,
                [hit.get("memory_id") for hit in hits if hit.get("memory_id")],
                edges,
                by_id,
            )
            member_ids = [anchor.id]
            for related_id in related_ids:
                pair = tuple(sorted((anchor.id, related_id)))
                if related_id != anchor.id and pair not in seen_pairs:
                    seen_pairs.add(pair)
                    member_ids.append(related_id)
                if len(member_ids) >= MAX_REVIEW_GROUP_MEMORIES:
                    break
            if len(member_ids) > 1:
                candidate = {
                    "anchor_id": anchor.id,
                    "scope_precedence": ["thread", "project", "user"],
                    "memories": [{
                        "id": by_id[item].id,
                        "scope_type": by_id[item].scope_type,
                        "scope_id": by_id[item].scope_id,
                        "scope_rank": SCOPE_RANK[by_id[item].scope_type],
                        "content": by_id[item].content,
                        "updated_at": iso_utc_z(by_id[item].updated_at or by_id[item].created_at),
                    } for item in member_ids],
                    "override_edges": _review_override_edges(member_ids, edges),
                }
                candidate_ids = {item["id"] for item in candidate["memories"]}
                overlapping = next((
                    group for group in groups
                    if candidate_ids.intersection(item["id"] for item in group["memories"])
                ), None)
                if overlapping is None:
                    candidate_size = len(json.dumps(candidate, ensure_ascii=True))
                    if groups and group_chars + candidate_size > review_budget:
                        break
                    groups.append(candidate)
                    group_chars += candidate_size
                else:
                    existing_ids = {item["id"] for item in overlapping["memories"]}
                    overlapping["memories"].extend(
                        item for item in candidate["memories"]
                        if item["id"] not in existing_ids
                    )
                    overlapping["memories"] = overlapping["memories"][:MAX_REVIEW_GROUP_MEMORIES]
                    overlapping["override_edges"] = _review_override_edges(
                        [item["id"] for item in overlapping["memories"]],
                        edges,
                    )
            processed_anchor_count += 1
    except Exception:
        logger.exception(
            "Memory consistency similarity search failed | context=%s:%s model=%s",
            context_type,
            context_id,
            model,
        )
        raise
    next_position = start + processed_anchor_count
    return {
        "context_type": context_type,
        "context_id": context_id,
        "snapshot_at": iso_utc_z(cutoff),
        "snapshot_scope_versions": snapshot_scope_versions or status["current_scope_versions"],
        "anchor_position": next_position,
        "reviewed_anchor_count": next_position,
        "remaining_anchor_count": max(0, len(anchors) - next_position),
        "candidate_groups": groups,
        "representation_pending": representation_pending,
        "missing_representation_count": len(missing_global_ids),
        "embedding_model": model,
        "blocked": False,
    }
