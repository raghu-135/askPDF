"""Repository for research-canvas artifacts."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.connection_sqlmodel import async_session_maker
from app.db.models_sqlmodel import AgentTaskArtifact
from app.models.canvas import RESEARCH_CANVAS_ARTIFACT_KIND


class CanvasRepository:
    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def _get_session(self) -> AsyncSession:
        if self._session is not None:
            return self._session
        return async_session_maker()

    async def create(self, artifact: AgentTaskArtifact) -> AgentTaskArtifact:
        session = await self._get_session()
        async with session.begin():
            session.add(artifact)
            await session.flush()
            await session.refresh(artifact)
            session.expunge(artifact)
            return artifact

    async def get(self, canvas_id: str) -> Optional[AgentTaskArtifact]:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(AgentTaskArtifact).where(
                    AgentTaskArtifact.id == canvas_id,
                    AgentTaskArtifact.kind == RESEARCH_CANVAS_ARTIFACT_KIND,
                    AgentTaskArtifact.validity != "deleted",
                )
            )
            artifact = result.scalar_one_or_none()
            if artifact is not None:
                session.expunge(artifact)
            return artifact

    async def get_by_idempotency(self, thread_id: str, idempotency_key: str) -> Optional[AgentTaskArtifact]:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(AgentTaskArtifact).where(
                    AgentTaskArtifact.thread_id == thread_id,
                    AgentTaskArtifact.kind == RESEARCH_CANVAS_ARTIFACT_KIND,
                    AgentTaskArtifact.idempotency_key == idempotency_key,
                    AgentTaskArtifact.validity != "deleted",
                )
            )
            artifact = result.scalar_one_or_none()
            if artifact is not None:
                session.expunge(artifact)
            return artifact

    async def list_for_thread(self, thread_id: str, *, current_only: bool = True) -> list[AgentTaskArtifact]:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(AgentTaskArtifact)
                .where(
                    AgentTaskArtifact.thread_id == thread_id,
                    AgentTaskArtifact.kind == RESEARCH_CANVAS_ARTIFACT_KIND,
                    AgentTaskArtifact.validity != "deleted",
                )
                .order_by(AgentTaskArtifact.created_at.desc(), AgentTaskArtifact.id.desc())
            )
            rows = list(result.scalars().all())
            for row in rows:
                session.expunge(row)
        if not current_only:
            return rows
        superseded = {row.supersedes_id for row in rows if row.supersedes_id}
        return [row for row in rows if row.id not in superseded]
