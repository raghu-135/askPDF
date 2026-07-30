"""Memory persistence operations with SQLModel."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, or_
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import MemoryScopeType, MemoryStatus, MemoryType, MemoryVisibility
from app.db.models_sqlmodel import Memory, MemoryEvent
from app.db.project_activity import touch_project_activity
from app.time_utils import utc_now


VALID_MEMORY_SCOPE_TYPES = {item.value for item in MemoryScopeType}
VALID_MEMORY_TYPES = {item.value for item in MemoryType}
VALID_MEMORY_STATUSES = {item.value for item in MemoryStatus}
VALID_MEMORY_VISIBILITIES = {item.value for item in MemoryVisibility}


def _require_nonempty(value: str, field_name: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} is required")
    return normalized


def _require_enum(value: str, field_name: str, allowed: set[str]) -> str:
    normalized = _require_nonempty(value, field_name)
    if normalized not in allowed:
        raise ValueError(f"invalid {field_name}: {normalized}")
    return normalized


def _normalize_confidence(value: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number between 0 and 1") from exc
    if confidence < 0 or confidence > 1:
        raise ValueError("confidence must be between 0 and 1")
    return confidence


class MemoryRepository:
    """Repository for canonical durable memories."""

    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def _get_session(self) -> AsyncSession:
        if self._session is not None:
            return self._session
        return async_session_maker()

    async def create_memory(
        self,
        *,
        scope_type: str,
        scope_id: str,
        memory_type: str,
        content: str,
        embedding_model: str,
        content_hash: str,
        summary: str = "",
        source_refs_json: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        visibility: str = "private",
        created_by: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        fork_origin_json: Optional[Dict[str, Any]] = None,
        event_payload: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        scope_type = _require_enum(scope_type, "scope_type", VALID_MEMORY_SCOPE_TYPES)
        scope_id = _require_nonempty(scope_id, "scope_id")
        memory_type = _require_enum(memory_type, "memory_type", VALID_MEMORY_TYPES)
        content = _require_nonempty(content, "content")
        visibility = _require_enum(visibility, "visibility", VALID_MEMORY_VISIBILITIES)
        confidence = _normalize_confidence(confidence)
        memory = Memory(
            scope_type=scope_type,
            scope_id=scope_id,
            memory_type=memory_type,
            content=content,
            summary=summary or "",
            embedding_model=_require_nonempty(embedding_model, "embedding_model"),
            content_hash=_require_nonempty(content_hash, "content_hash"),
            index_status="pending",
            source_refs_json=source_refs_json or {},
            confidence=confidence,
            visibility=visibility,
            created_by=created_by,
            expires_at=expires_at,
            fork_origin_json=fork_origin_json,
            created_at=utc_now(),
        )
        session = await self._get_session()
        async with session.begin():
            session.add(memory)
            await session.flush()
            session.add(
                MemoryEvent(
                    memory_id=memory.id,
                    event_type="created",
                    actor_id=created_by,
                    payload_json=event_payload or {},
                    created_at=utc_now(),
                )
            )
            await session.flush()
            if scope_type == MemoryScopeType.PROJECT.value:
                await touch_project_activity(
                    session,
                    scope_id,
                    occurred_at=memory.created_at,
                )
            await session.refresh(memory)
            return memory

    async def get_memory(self, memory_id: str) -> Optional[Memory]:
        session = await self._get_session()
        async with session.begin():
            return await session.get(Memory, memory_id)

    async def list_memories(
        self,
        *,
        scope_type: Optional[str] = None,
        scope_id: Optional[str] = None,
        status: str = MemoryStatus.ACTIVE.value,
        limit: int = 100,
    ) -> list[Memory]:
        bounded_limit = max(1, min(int(limit), 500))
        if scope_type is not None:
            scope_type = _require_enum(scope_type, "scope_type", VALID_MEMORY_SCOPE_TYPES)
        if scope_id is not None:
            scope_id = _require_nonempty(scope_id, "scope_id")
        if status:
            status = _require_enum(status, "status", VALID_MEMORY_STATUSES)
        session = await self._get_session()
        async with session.begin():
            query = select(Memory)
            if scope_type is not None:
                query = query.where(Memory.scope_type == scope_type)
            if scope_id is not None:
                query = query.where(Memory.scope_id == scope_id)
            if status:
                query = query.where(Memory.status == status)
            query = query.where(
                or_(Memory.expires_at.is_(None), Memory.expires_at > utc_now())
            )
            result = await session.execute(
                query.order_by(Memory.updated_at.desc().nullslast(), Memory.created_at.desc()).limit(bounded_limit)
            )
            return list(result.scalars().all())

    async def mark_memory_indexing(self, memory_id: str) -> Optional[Memory]:
        session = await self._get_session()
        async with session.begin():
            memory = await session.get(Memory, memory_id)
            if memory is None:
                return None
            memory.index_status = "indexing"
            memory.index_attempts = int(memory.index_attempts or 0) + 1
            memory.index_error = None
            memory.updated_at = utc_now()
            await session.flush()
            await session.refresh(memory)
            return memory

    async def mark_memory_indexed(self, memory_id: str) -> Optional[Memory]:
        session = await self._get_session()
        async with session.begin():
            memory = await session.get(Memory, memory_id)
            if memory is None:
                return None
            now = utc_now()
            memory.index_status = "indexed"
            memory.indexed_at = now
            memory.index_error = None
            memory.updated_at = now
            await session.flush()
            await session.refresh(memory)
            return memory

    async def mark_memory_index_failed(self, memory_id: str, error: str) -> Optional[Memory]:
        session = await self._get_session()
        async with session.begin():
            memory = await session.get(Memory, memory_id)
            if memory is None:
                return None
            memory.index_status = "failed"
            memory.index_error = str(error or "Memory indexing failed")[:2000]
            memory.updated_at = utc_now()
            await session.flush()
            await session.refresh(memory)
            return memory

    async def update_memory(
        self,
        memory_id: str,
        *,
        memory_type: str,
        content: str,
        content_hash: str,
        summary: str = "",
        confidence: float = 1.0,
        source_refs_json: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        event_type: str = "updated",
        event_payload: Optional[Dict[str, Any]] = None,
        updated_at: Optional[datetime] = None,
    ) -> Optional[Memory]:
        """Update one canonical row, reusing an injected transaction when present."""

        memory_id = _require_nonempty(memory_id, "memory_id")
        memory_type = _require_enum(memory_type, "memory_type", VALID_MEMORY_TYPES)
        content = _require_nonempty(content, "content")
        content_hash = _require_nonempty(content_hash, "content_hash")
        confidence = _normalize_confidence(confidence)

        async def apply(session: AsyncSession) -> Optional[Memory]:
            memory = await session.get(Memory, memory_id)
            if memory is None:
                return None
            now = updated_at or utc_now()
            memory.memory_type = memory_type
            memory.content = content
            memory.summary = summary or ""
            memory.content_hash = content_hash
            memory.confidence = confidence
            memory.source_refs_json = {
                **dict(memory.source_refs_json or {}),
                **dict(source_refs_json or {}),
            }
            memory.index_status = "pending"
            memory.indexed_at = None
            memory.index_error = None
            memory.updated_at = now
            session.add(MemoryEvent(
                memory_id=memory.id,
                event_type=_require_nonempty(event_type, "event_type"),
                actor_id=actor_id,
                payload_json=event_payload or {},
                created_at=now,
            ))
            await session.flush()
            if memory.scope_type == MemoryScopeType.PROJECT.value:
                await touch_project_activity(session, memory.scope_id, occurred_at=now)
            return memory

        session = await self._get_session()
        if self._session is not None:
            return await apply(session)
        async with session.begin():
            return await apply(session)

    async def list_memories_for_index_retry(self, *, limit: int = 100) -> list[Memory]:
        bounded_limit = max(1, min(int(limit), 500))
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Memory)
                .where(
                    Memory.status == MemoryStatus.ACTIVE.value,
                    Memory.index_status.in_(("pending", "failed")),
                    or_(Memory.expires_at.is_(None), Memory.expires_at > utc_now()),
                )
                .order_by(Memory.updated_at.asc().nullsfirst(), Memory.created_at.asc())
                .limit(bounded_limit)
            )
            return list(result.scalars().all())

    async def delete_memory(self, memory_id: str) -> bool:
        memory_id = _require_nonempty(memory_id, "memory_id")
        session = await self._get_session()
        async with session.begin():
            memory = await session.get(Memory, memory_id)
            if memory is None:
                return False
            project_id = (
                memory.scope_id
                if memory.scope_type == MemoryScopeType.PROJECT.value
                else None
            )
            await session.execute(delete(MemoryEvent).where(MemoryEvent.memory_id == memory_id))
            await session.delete(memory)
            await session.flush()
            if project_id:
                await touch_project_activity(session, project_id)
            return True

    async def delete_memories_for_scope(self, *, scope_type: str, scope_id: str) -> list[str]:
        scope_type = _require_enum(scope_type, "scope_type", VALID_MEMORY_SCOPE_TYPES)
        scope_id = _require_nonempty(scope_id, "scope_id")
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Memory.id).where(Memory.scope_type == scope_type, Memory.scope_id == scope_id)
            )
            memory_ids = [str(row[0]) for row in result.all()]
            if not memory_ids:
                return []
            await session.execute(delete(MemoryEvent).where(MemoryEvent.memory_id.in_(memory_ids)))
            await session.execute(delete(Memory).where(Memory.id.in_(memory_ids)))
            return memory_ids

    async def delete_expired_memories(self, *, now: Optional[datetime] = None) -> list[Memory]:
        cutoff = now or utc_now()
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Memory).where(Memory.expires_at.is_not(None), Memory.expires_at <= cutoff)
            )
            memories = list(result.scalars().all())
            if not memories:
                return []
            memory_ids = [memory.id for memory in memories]
            await session.execute(delete(MemoryEvent).where(MemoryEvent.memory_id.in_(memory_ids)))
            await session.execute(delete(Memory).where(Memory.id.in_(memory_ids)))
            return memories

    async def list_expired_memories(self, *, now: Optional[datetime] = None, limit: int = 500) -> list[Memory]:
        cutoff = now or utc_now()
        bounded_limit = max(1, min(int(limit), 500))
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Memory)
                .where(Memory.expires_at.is_not(None), Memory.expires_at <= cutoff)
                .order_by(Memory.expires_at.asc())
                .limit(bounded_limit)
            )
            return list(result.scalars().all())
