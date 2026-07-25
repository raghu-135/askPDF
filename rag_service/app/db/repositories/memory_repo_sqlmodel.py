"""Memory persistence operations with SQLModel."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import delete, or_
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import MemoryCandidateStatus, MemoryScopeType, MemoryStatus, MemoryType, MemoryVisibility
from app.db.models_sqlmodel import Memory, MemoryCandidate, MemoryEvent
from app.time_utils import utc_now


VALID_MEMORY_SCOPE_TYPES = {item.value for item in MemoryScopeType}
VALID_MEMORY_TYPES = {item.value for item in MemoryType}
VALID_MEMORY_STATUSES = {item.value for item in MemoryStatus}
VALID_MEMORY_VISIBILITIES = {item.value for item in MemoryVisibility}
VALID_CANDIDATE_STATUSES = {item.value for item in MemoryCandidateStatus}


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
    """Repository for canonical durable memories and promotion candidates."""

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
            await session.execute(delete(MemoryEvent).where(MemoryEvent.memory_id == memory_id))
            await session.delete(memory)
            await session.flush()
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

    async def create_candidate(
        self,
        *,
        proposed_scope_type: str,
        proposed_scope_id: str,
        memory_type: str,
        content: str,
        source_thread_id: Optional[str] = None,
        source_project_id: Optional[str] = None,
        source_agent_run_id: Optional[str] = None,
        source_turn_id: Optional[str] = None,
        confidence: float = 0.0,
        reason: str = "",
        created_by: Optional[str] = None,
    ) -> MemoryCandidate:
        proposed_scope_type = _require_enum(proposed_scope_type, "proposed_scope_type", VALID_MEMORY_SCOPE_TYPES)
        proposed_scope_id = _require_nonempty(proposed_scope_id, "proposed_scope_id")
        memory_type = _require_enum(memory_type, "memory_type", VALID_MEMORY_TYPES)
        content = _require_nonempty(content, "content")
        confidence = _normalize_confidence(confidence)
        candidate = MemoryCandidate(
            source_thread_id=source_thread_id,
            source_project_id=source_project_id,
            source_agent_run_id=source_agent_run_id,
            source_turn_id=source_turn_id,
            proposed_scope_type=proposed_scope_type,
            proposed_scope_id=proposed_scope_id,
            memory_type=memory_type,
            content=content,
            confidence=confidence,
            reason=reason or "",
            created_by=created_by,
            created_at=utc_now(),
        )
        session = await self._get_session()
        async with session.begin():
            session.add(candidate)
            await session.flush()
            await session.refresh(candidate)
            return candidate

    async def list_candidates(
        self,
        *,
        status: str = MemoryCandidateStatus.PENDING.value,
        source_project_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[MemoryCandidate]:
        bounded_limit = max(1, min(int(limit), 500))
        if status:
            status = _require_enum(status, "status", VALID_CANDIDATE_STATUSES)
        session = await self._get_session()
        async with session.begin():
            query = select(MemoryCandidate)
            if status:
                query = query.where(MemoryCandidate.status == status)
            if source_project_id is not None:
                query = query.where(MemoryCandidate.source_project_id == source_project_id)
            result = await session.execute(query.order_by(MemoryCandidate.created_at.desc()).limit(bounded_limit))
            return list(result.scalars().all())

    async def get_candidate(self, candidate_id: str) -> Optional[MemoryCandidate]:
        session = await self._get_session()
        async with session.begin():
            return await session.get(MemoryCandidate, candidate_id)

    async def resolve_candidate(
        self,
        candidate_id: str,
        *,
        status: str,
        actor_id: Optional[str] = None,
    ) -> Optional[MemoryCandidate]:
        status = _require_enum(status, "status", VALID_CANDIDATE_STATUSES)
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(MemoryCandidate)
                .where(MemoryCandidate.id == candidate_id)
                .with_for_update()
            )
            candidate = result.scalar_one_or_none()
            if candidate is None:
                return None
            if candidate.status != MemoryCandidateStatus.PENDING.value:
                if candidate.status != status:
                    raise ValueError(
                        f"Memory candidate is already {candidate.status}"
                    )
                return candidate
            candidate.status = status
            candidate.resolved_by = actor_id
            candidate.resolved_at = utc_now()
            await session.flush()
            await session.refresh(candidate)
            return candidate

    async def promote_candidate(
        self,
        candidate_id: str,
        *,
        status: str,
        embedding_model: str,
        content_hash: str,
        actor_id: Optional[str] = None,
    ) -> tuple[Optional[MemoryCandidate], Optional[Memory], bool]:
        """Atomically resolve a pending candidate and create its canonical memory once."""

        if status not in {
            MemoryCandidateStatus.APPROVED.value,
            MemoryCandidateStatus.AUTO_APPROVED.value,
        }:
            raise ValueError("Candidate promotion requires an approved status")
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(MemoryCandidate)
                .where(MemoryCandidate.id == candidate_id)
                .with_for_update()
            )
            candidate = result.scalar_one_or_none()
            if candidate is None:
                return None, None, False
            if candidate.status != MemoryCandidateStatus.PENDING.value:
                if candidate.status not in {
                    MemoryCandidateStatus.APPROVED.value,
                    MemoryCandidateStatus.AUTO_APPROVED.value,
                }:
                    raise ValueError(
                        f"Memory candidate is already {candidate.status}"
                    )
                memory = (
                    await session.get(Memory, candidate.promoted_memory_id)
                    if candidate.promoted_memory_id
                    else None
                )
                return candidate, memory, False

            memory = Memory(
                scope_type=candidate.proposed_scope_type,
                scope_id=candidate.proposed_scope_id,
                memory_type=candidate.memory_type,
                content=candidate.content,
                summary="",
                embedding_model=_require_nonempty(embedding_model, "embedding_model"),
                content_hash=_require_nonempty(content_hash, "content_hash"),
                index_status="pending",
                source_refs_json={
                    "candidate_id": candidate.id,
                    "source_thread_id": candidate.source_thread_id,
                    "source_project_id": candidate.source_project_id,
                    "source_agent_run_id": candidate.source_agent_run_id,
                    "source_turn_id": candidate.source_turn_id,
                },
                confidence=candidate.confidence,
                visibility=(
                    MemoryVisibility.PROJECT.value
                    if candidate.proposed_scope_type == MemoryScopeType.PROJECT.value
                    else MemoryVisibility.PRIVATE.value
                ),
                created_by=actor_id or candidate.created_by,
                created_at=utc_now(),
            )
            session.add(memory)
            await session.flush()
            session.add(
                MemoryEvent(
                    memory_id=memory.id,
                    event_type="created",
                    actor_id=actor_id or candidate.created_by,
                    payload_json={"candidate_id": candidate.id},
                    created_at=utc_now(),
                )
            )
            candidate.status = status
            candidate.promoted_memory_id = memory.id
            candidate.resolved_by = actor_id
            candidate.resolved_at = utc_now()
            await session.flush()
            await session.refresh(candidate)
            await session.refresh(memory)
            return candidate, memory, True

    async def delete_candidate(self, candidate_id: str) -> bool:
        candidate_id = _require_nonempty(candidate_id, "candidate_id")
        session = await self._get_session()
        async with session.begin():
            candidate = await session.get(MemoryCandidate, candidate_id)
            if candidate is None:
                return False
            await session.delete(candidate)
            await session.flush()
            return True

    async def delete_candidates_for_thread(self, thread_id: str) -> list[str]:
        thread_id = _require_nonempty(thread_id, "thread_id")
        session = await self._get_session()
        async with session.begin():
            query = select(MemoryCandidate.id).where(
                or_(
                    MemoryCandidate.source_thread_id == thread_id,
                    (
                        (MemoryCandidate.proposed_scope_type == MemoryScopeType.THREAD.value)
                        & (MemoryCandidate.proposed_scope_id == thread_id)
                    ),
                )
            )
            result = await session.execute(query)
            candidate_ids = [str(row[0]) for row in result.all()]
            if not candidate_ids:
                return []
            await session.execute(delete(MemoryCandidate).where(MemoryCandidate.id.in_(candidate_ids)))
            return candidate_ids
