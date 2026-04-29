"""
thread_repo_sqlmodel.py - Thread CRUD operations with SQLModel.

This module provides repository methods for managing thread entities
using SQLModel with PostgreSQL, including JSONB settings handling.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from app.db.models_sqlmodel import Thread, Message, ThreadFile
from app.db.jsonb_utils import merge_jsonb_field
from app.db.connection_sqlmodel import async_session_maker


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
        created_at = datetime.utcnow()

        thread = Thread(
            id=thread_id,
            name=name,
            embed_model=embed_model,
            settings={},
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
            # Query threads with counts using subqueries
            thread_result = await session.execute(
                select(
                    Thread,
                    func.count(func.distinct(Message.id)).label("message_count"),
                    func.count(func.distinct(ThreadFile.file_hash)).label("file_count"),
                    func.max(Message.created_at).label("last_message_at")
                )
                .outerjoin(Message, Thread.id == Message.thread_id)
                .outerjoin(ThreadFile, Thread.id == ThreadFile.thread_id)
                .group_by(Thread.id)
                .order_by(func.coalesce(func.max(Message.created_at), Thread.created_at).desc())
            )

            threads = []
            for row in thread_result.all():
                thread = row[0]
                threads.append({
                    "id": thread.id,
                    "name": thread.name,
                    "embed_model": thread.embed_model,
                    "settings": thread.settings if thread.settings else {},
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
