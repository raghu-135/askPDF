"""Shared project activity timestamp updates."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.models_sqlmodel import Project, Thread
from app.time_utils import utc_now


async def touch_project_activity(
    session: AsyncSession,
    project_id: str,
    *,
    occurred_at: Optional[datetime] = None,
) -> None:
    """Advance a project's activity timestamp without moving it backwards."""
    project = await session.get(Project, project_id)
    if project is None:
        return
    activity_at = occurred_at or utc_now()
    if project.last_activity_at is None or activity_at > project.last_activity_at:
        project.last_activity_at = activity_at


async def touch_thread_project_activity(
    session: AsyncSession,
    thread_id: str,
    *,
    occurred_at: Optional[datetime] = None,
) -> None:
    """Advance activity for the project that owns a thread."""
    result = await session.execute(
        select(Thread.project_id).where(Thread.id == thread_id)
    )
    project_id = result.scalar_one_or_none()
    if project_id:
        await touch_project_activity(
            session,
            project_id,
            occurred_at=occurred_at,
        )
