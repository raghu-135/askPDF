"""Repository for thread-owned research canvases."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.connection_sqlmodel import async_session_maker
from app.db.models_sqlmodel import ThreadCanvas


class CanvasRepository:
    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def _get_session(self) -> AsyncSession:
        if self._session is not None:
            return self._session
        return async_session_maker()

    async def create(self, canvas: ThreadCanvas) -> ThreadCanvas:
        session = await self._get_session()
        async with session.begin():
            session.add(canvas)
            await session.flush()
            await session.refresh(canvas)
            session.expunge(canvas)
            return canvas

    async def get(self, canvas_id: str) -> Optional[ThreadCanvas]:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(select(ThreadCanvas).where(ThreadCanvas.id == canvas_id))
            canvas = result.scalar_one_or_none()
            if canvas is not None:
                session.expunge(canvas)
            return canvas

    async def get_by_idempotency(self, thread_id: str, idempotency_key: str) -> Optional[ThreadCanvas]:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadCanvas).where(
                    ThreadCanvas.thread_id == thread_id,
                    ThreadCanvas.idempotency_key == idempotency_key,
                )
            )
            canvas = result.scalar_one_or_none()
            if canvas is not None:
                session.expunge(canvas)
            return canvas

    async def list_for_thread(self, thread_id: str, *, current_only: bool = True) -> list[ThreadCanvas]:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadCanvas)
                .where(ThreadCanvas.thread_id == thread_id)
                .order_by(ThreadCanvas.created_at.desc(), ThreadCanvas.id.desc())
            )
            rows = list(result.scalars().all())
            for row in rows:
                session.expunge(row)
        if not current_only:
            return rows
        superseded = {row.supersedes_id for row in rows if row.supersedes_id}
        return [row for row in rows if row.id not in superseded]
