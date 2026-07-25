"""Memory persistence operations with SQLModel."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import MemoryCandidateStatus, MemoryStatus
from app.db.models_sqlmodel import Memory, MemoryCandidate, MemoryEvent
from app.time_utils import utc_now


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
        summary: str = "",
        source_refs_json: Optional[Dict[str, Any]] = None,
        confidence: float = 1.0,
        visibility: str = "private",
        created_by: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        fork_origin_json: Optional[Dict[str, Any]] = None,
        event_payload: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        memory = Memory(
            scope_type=scope_type,
            scope_id=scope_id,
            memory_type=memory_type,
            content=content,
            summary=summary or "",
            source_refs_json=source_refs_json or {},
            confidence=float(confidence),
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
        session = await self._get_session()
        async with session.begin():
            query = select(Memory)
            if scope_type is not None:
                query = query.where(Memory.scope_type == scope_type)
            if scope_id is not None:
                query = query.where(Memory.scope_id == scope_id)
            if status:
                query = query.where(Memory.status == status)
            result = await session.execute(
                query.order_by(Memory.updated_at.desc().nullslast(), Memory.created_at.desc()).limit(bounded_limit)
            )
            return list(result.scalars().all())

    async def update_memory_status(
        self,
        memory_id: str,
        *,
        status: str,
        actor_id: Optional[str] = None,
        payload_json: Optional[Dict[str, Any]] = None,
    ) -> Optional[Memory]:
        session = await self._get_session()
        async with session.begin():
            memory = await session.get(Memory, memory_id)
            if memory is None:
                return None
            memory.status = status
            session.add(
                MemoryEvent(
                    memory_id=memory.id,
                    event_type=status,
                    actor_id=actor_id,
                    payload_json=payload_json or {},
                    created_at=utc_now(),
                )
            )
            await session.flush()
            await session.refresh(memory)
            return memory

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
        candidate = MemoryCandidate(
            source_thread_id=source_thread_id,
            source_project_id=source_project_id,
            source_agent_run_id=source_agent_run_id,
            source_turn_id=source_turn_id,
            proposed_scope_type=proposed_scope_type,
            proposed_scope_id=proposed_scope_id,
            memory_type=memory_type,
            content=content,
            confidence=float(confidence),
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

    async def resolve_candidate(self, candidate_id: str, *, status: str) -> Optional[MemoryCandidate]:
        session = await self._get_session()
        async with session.begin():
            candidate = await session.get(MemoryCandidate, candidate_id)
            if candidate is None:
                return None
            candidate.status = status
            await session.flush()
            await session.refresh(candidate)
            return candidate
