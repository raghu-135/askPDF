"""
thread_repo_sqlmodel.py - Thread CRUD operations with SQLModel.

This module provides repository methods for managing thread entities
using SQLModel with PostgreSQL, including JSONB settings handling.
"""

import uuid
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from app.db.models_sqlmodel import Thread, ChatTurn, ThreadFile
from app.db.jsonb_utils import merge_jsonb_field
from app.db.connection_sqlmodel import async_session_maker
from app.time_utils import utc_now


class ThreadRepository:
    """Repository for thread database operations using SQLModel."""

    def __init__(self, session: Optional[AsyncSession] = None):
        """Initialize with optional session for dependency injection."""
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get a database session - injected for tests, default for production."""
        if self._session is not None:
            return self._session
        return async_session_maker()

    async def create(self, name: str, embed_model: str) -> Thread:
        """Create a new thread with default settings."""
        thread_id = str(uuid.uuid4())
        created_at = utc_now()

        thread = Thread(
            id=thread_id,
            name=name,
            embed_model=embed_model,
            settings={},
            thread_metadata={},
            created_at=created_at
        )

        session = await self._get_session()
        async with session.begin():
            session.add(thread)
            await session.flush()
            await session.refresh(thread)
        return thread

    async def get(self, thread_id: str) -> Optional[Thread]:
        """Get a thread by ID."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Thread).where(Thread.id == thread_id)
            )
            return result.scalar_one_or_none()

    async def get_settings(self, thread_id: str) -> Dict[str, Any]:
        """Get persisted settings for a thread."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Thread.settings).where(Thread.id == thread_id)
            )
            settings = result.scalar_one_or_none()
            return settings if settings else {}

    async def update_settings(
        self,
        thread_id: str,
        settings: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Replace persisted settings for a thread with proper JSONB tracking."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Thread).where(Thread.id == thread_id)
            )
            thread = result.scalar_one_or_none()
            if not thread:
                return None

            # Use JSONB utility for safe mutation with change tracking
            merge_jsonb_field(thread, "settings", settings or {})
            await session.flush()
            return thread.settings

    async def list_all(self) -> List[Dict[str, Any]]:
        """List all threads with message counts and file counts."""
        session = await self._get_session()
        async with session.begin():
            question_count = (
                select(func.count(ChatTurn.id))
                .where(
                    ChatTurn.thread_id == Thread.id,
                    ChatTurn.status != "cancelled",
                    ChatTurn.payload["question"].astext.isnot(None),
                    ChatTurn.payload["question"].astext != "",
                )
                .correlate(Thread)
                .scalar_subquery()
            )
            answer_count = (
                select(func.count(ChatTurn.id))
                .where(
                    ChatTurn.thread_id == Thread.id,
                    ChatTurn.status != "cancelled",
                    ChatTurn.payload["answer"].astext.isnot(None),
                    ChatTurn.payload["answer"].astext != "",
                )
                .correlate(Thread)
                .scalar_subquery()
            )
            file_count = (
                select(func.count(ThreadFile.file_hash))
                .where(ThreadFile.thread_id == Thread.id)
                .correlate(Thread)
                .scalar_subquery()
            )
            last_message_at = (
                select(func.max(ChatTurn.created_at))
                .where(ChatTurn.thread_id == Thread.id, ChatTurn.status != "cancelled")
                .correlate(Thread)
                .scalar_subquery()
            )
            thread_result = await session.execute(
                select(
                    Thread,
                    (question_count + answer_count).label("message_count"),
                    file_count.label("file_count"),
                    last_message_at.label("last_message_at")
                )
                .order_by(func.coalesce(last_message_at, Thread.created_at).desc())
            )

            threads = []
            for row in thread_result.all():
                thread = row[0]
                threads.append({
                    "id": thread.id,
                    "name": thread.name,
                    "embed_model": thread.embed_model,
                    "settings": thread.settings if thread.settings else {},
                    "thread_metadata": thread.thread_metadata if thread.thread_metadata else {},
                    "created_at": thread.created_at,
                    "message_count": row[1] or 0,
                    "file_count": row[2] or 0
                })

            return threads

    async def update(self, thread_id: str, name: str) -> Optional[Thread]:
        """Update a thread's name."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Thread).where(Thread.id == thread_id)
            )
            thread = result.scalar_one_or_none()
            if not thread:
                return None

            thread.name = name
            await session.flush()
            await session.refresh(thread)
        return thread

    async def delete(self, thread_id: str) -> bool:
        """Delete a thread and all associated data (cascade handled by DB)."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Thread).where(Thread.id == thread_id)
            )
            thread = result.scalar_one_or_none()
            if not thread:
                return False

            await session.delete(thread)
        return True
