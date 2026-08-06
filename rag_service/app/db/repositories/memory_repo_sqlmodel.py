"""Memory persistence operations with SQLModel."""

from __future__ import annotations

from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import MemoryScopeType
from app.db.models_sqlmodel import Memory, MemoryEvent, MemoryOverride
from app.db.project_activity import touch_project_activity
from app.models.memory_tools import normalize_memory_attributes
from app.models.memory_limits import MAX_MEMORY_ROWS
from app.time_utils import utc_now


VALID_MEMORY_SCOPE_TYPES = {item.value for item in MemoryScopeType}


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
        content: str,
        embedding_model: str,
        content_hash: str,
        source_refs_json: Optional[Dict[str, Any]] = None,
        attributes_json: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        event_payload: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        scope_type = _require_enum(scope_type, "scope_type", VALID_MEMORY_SCOPE_TYPES)
        scope_id = _require_nonempty(scope_id, "scope_id")
        content = _require_nonempty(content, "content")
        memory = Memory(
            scope_type=scope_type,
            scope_id=scope_id,
            content=content,
            embedding_model=_require_nonempty(embedding_model, "embedding_model"),
            content_hash=_require_nonempty(content_hash, "content_hash"),
            index_status="pending",
            source_refs_json=source_refs_json or {},
            attributes_json=normalize_memory_attributes(attributes_json),
            created_at=utc_now(),
        )
        session = await self._get_session()
        async with session.begin():
            session.add(memory)
            await session.flush()
            from app.services.memory_review_service import bump_memory_scope_activity
            from app.services.memory_representation_service import invalidate_global_representations
            await bump_memory_scope_activity([(scope_type, scope_id)], session=session)
            await invalidate_global_representations(memory, session=session)
            session.add(
                MemoryEvent(
                    memory_id=memory.id,
                    event_type="created",
                    actor_id=actor_id,
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
        limit: int = 100,
    ) -> list[Memory]:
        bounded_limit = max(1, min(int(limit), MAX_MEMORY_ROWS))
        if scope_type is not None:
            scope_type = _require_enum(scope_type, "scope_type", VALID_MEMORY_SCOPE_TYPES)
        if scope_id is not None:
            scope_id = _require_nonempty(scope_id, "scope_id")
        session = await self._get_session()
        async with session.begin():
            query = select(Memory)
            if scope_type is not None:
                query = query.where(Memory.scope_type == scope_type)
            if scope_id is not None:
                query = query.where(Memory.scope_id == scope_id)
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
        content: str,
        content_hash: str,
        source_refs_json: Optional[Dict[str, Any]] = None,
        attributes_json: Optional[Dict[str, Any]] = None,
        actor_id: Optional[str] = None,
        event_type: str = "updated",
        event_payload: Optional[Dict[str, Any]] = None,
        updated_at: Optional[datetime] = None,
    ) -> Optional[Memory]:
        """Update one canonical row, reusing an injected transaction when present."""

        memory_id = _require_nonempty(memory_id, "memory_id")
        content = _require_nonempty(content, "content")
        content_hash = _require_nonempty(content_hash, "content_hash")

        async def apply(session: AsyncSession) -> Optional[Memory]:
            memory = await session.get(Memory, memory_id)
            if memory is None:
                return None
            now = updated_at or utc_now()
            memory.content = content
            memory.content_hash = content_hash
            memory.source_refs_json = {
                **dict(memory.source_refs_json or {}),
                **dict(source_refs_json or {}),
            }
            if attributes_json is not None:
                memory.attributes_json = normalize_memory_attributes(attributes_json)
            memory.index_status = "pending"
            memory.indexed_at = None
            memory.index_error = None
            memory.updated_at = now
            memory.semantic_updated_at = now
            session.add(MemoryEvent(
                memory_id=memory.id,
                event_type=_require_nonempty(event_type, "event_type"),
                actor_id=actor_id,
                payload_json=event_payload or {},
                created_at=now,
            ))
            await session.flush()
            from app.services.memory_review_service import bump_memory_scope_activity
            from app.services.memory_representation_service import invalidate_global_representations
            await bump_memory_scope_activity([(memory.scope_type, memory.scope_id)], session=session)
            await invalidate_global_representations(memory, session=session)
            if memory.scope_type == MemoryScopeType.PROJECT.value:
                await touch_project_activity(session, memory.scope_id, occurred_at=now)
            return memory

        session = await self._get_session()
        if self._session is not None:
            return await apply(session)
        async with session.begin():
            return await apply(session)

    async def list_memories_for_index_retry(self, *, limit: int = 100) -> list[Memory]:
        bounded_limit = max(1, min(int(limit), MAX_MEMORY_ROWS))
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Memory)
                .where(
                    Memory.index_status.in_(("pending", "failed")),
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
            from app.services.memory_review_service import bump_memory_scope_activity
            await bump_memory_scope_activity([(memory.scope_type, memory.scope_id)], session=session)
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
            from app.services.memory_review_service import bump_memory_scope_activity
            await bump_memory_scope_activity([(scope_type, scope_id)], session=session)
            await session.execute(delete(MemoryEvent).where(MemoryEvent.memory_id.in_(memory_ids)))
            await session.execute(delete(Memory).where(Memory.id.in_(memory_ids)))
            return memory_ids

    async def list_override_edges(self, *, memory_ids: Optional[list[str]] = None) -> list[MemoryOverride]:
        session = await self._get_session()
        async with session.begin():
            query = select(MemoryOverride)
            if memory_ids is not None:
                if not memory_ids:
                    return []
                query = query.where(
                    (MemoryOverride.overriding_memory_id.in_(memory_ids))
                    | (MemoryOverride.overridden_memory_id.in_(memory_ids))
                )
            result = await session.execute(query.order_by(MemoryOverride.created_at))
            return list(result.scalars().all())

    async def replace_overrides(
        self,
        memory_id: str,
        target_ids: list[str],
        *,
        actor_id: Optional[str] = None,
        updated_at=None,
    ) -> None:
        """Replace all outgoing override edges in an existing transaction."""

        if self._session is None:
            raise RuntimeError("replace_overrides requires an injected transaction")
        memory_id = _require_nonempty(memory_id, "memory_id")
        normalized_targets = sorted({_require_nonempty(item, "overridden_memory_id") for item in target_ids})
        memory = await self._session.get(Memory, memory_id)
        if memory is None:
            raise ValueError(f"memory not found: {memory_id}")
        existing_target_ids = list((await self._session.execute(
            select(MemoryOverride.overridden_memory_id).where(
                MemoryOverride.overriding_memory_id == memory_id
            )
        )).scalars().all())
        affected_ids = sorted(set(existing_target_ids) | set(normalized_targets))
        affected_memories = list((await self._session.execute(
            select(Memory).where(Memory.id.in_(affected_ids))
        )).scalars().all()) if affected_ids else []
        await self._session.execute(
            delete(MemoryOverride).where(MemoryOverride.overriding_memory_id == memory_id)
        )
        now = updated_at or utc_now()
        memory.updated_at = now
        for target_id in normalized_targets:
            self._session.add(MemoryOverride(
                overriding_memory_id=memory_id,
                overridden_memory_id=target_id,
                created_at=now,
            ))
        self._session.add(MemoryEvent(
            memory_id=memory_id,
            event_type="override_set",
            actor_id=actor_id,
            payload_json={"overridden_memory_ids": normalized_targets},
            created_at=now,
        ))
        from app.services.memory_review_service import bump_memory_scope_activity
        await bump_memory_scope_activity(
            [(memory.scope_type, memory.scope_id)]
            + [(target.scope_type, target.scope_id) for target in affected_memories],
            session=self._session,
        )
        await self._session.flush()
