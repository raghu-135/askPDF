"""
message_repo_sqlmodel.py - Message CRUD operations with SQLModel.

This module provides repository methods for managing message entities
using SQLModel with PostgreSQL, including web_sources JSONB handling.
"""

import uuid
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.models_sqlmodel import Message, MessageRole, Thread
from app.db.jsonb_utils import merge_jsonb_field
from app.db.connection_sqlmodel import async_session_maker


class MessageRepository:
    """Repository for message database operations using SQLModel."""

    def __init__(self, session: Optional[AsyncSession] = None):
        """Initialize with optional session for dependency injection."""
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get a database session - injected for tests, default for production."""
        if self._session is not None:
            return self._session
        return async_session_maker()

    async def create(
        self,
        thread_id: str,
        role: MessageRole,
        content: str,
        context_compact: Optional[str] = None,
        reasoning: Optional[str] = None,
        reasoning_available: bool = False,
        reasoning_format: str = "none",
        web_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        """Create a new message in a thread."""
        message_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        message = Message(
            id=message_id,
            thread_id=thread_id,
            role=role.value if isinstance(role, MessageRole) else role,
            content=content,
            context_compact=context_compact,
            reasoning=reasoning,
            reasoning_available=reasoning_available,
            reasoning_format=reasoning_format,
            web_sources=web_sources,
            created_at=created_at,
        )

        session = await self._get_session()
        async with session.begin():
            session.add(message)
            await session.flush()
            await session.commit()
            await session.refresh(message)
            return message

    async def get(self, message_id: str) -> Optional[Message]:
        """Get a message by ID."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Message).where(Message.id == message_id)
            )
            return result.scalar_one_or_none()

    async def get_thread_messages(
        self,
        thread_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a thread with pagination."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Message)
                .where(Message.thread_id == thread_id)
                .order_by(Message.created_at.asc())
                .limit(limit)
                .offset(offset)
            )
            return list(result.scalars().all())

    async def get_recent_messages(
        self,
        thread_id: str,
        limit: int = 10
    ) -> List[Message]:
        """Get the most recent messages for a thread (for context window)."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Message)
                .where(Message.thread_id == thread_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
            # Reverse to get chronological order
            messages = list(result.scalars().all())
            return list(reversed(messages))

    async def update_context_compact(
        self,
        message_id: str,
        context_compact: str
    ) -> bool:
        """Update compact context text for a message."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Message).where(Message.id == message_id)
            )
            message = result.scalar_one_or_none()
            if not message:
                return False

            message.context_compact = context_compact
            await session.flush()
            await session.commit()
            return True

    async def update_reasoning(
        self,
        message_id: str,
        reasoning: str,
        reasoning_format: str = "raw",
    ) -> bool:
        """Update reasoning for a message."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Message).where(Message.id == message_id)
            )
            message = result.scalar_one_or_none()
            if not message:
                return False

            message.reasoning = reasoning
            message.reasoning_available = True
            message.reasoning_format = reasoning_format
            await session.flush()
            await session.commit()
            return True

    async def update_web_sources(
        self,
        message_id: str,
        web_sources: List[Dict[str, Any]]
    ) -> bool:
        """Update web sources for a message with proper JSONB tracking."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Message).where(Message.id == message_id)
            )
            message = result.scalar_one_or_none()
            if not message:
                return False

            # Use JSONB utility for safe mutation with change tracking
            merge_jsonb_field(message, "web_sources", web_sources)
            await session.flush()
            await session.commit()
            return True

    async def delete(self, message_id: str) -> bool:
        """Delete a message by ID."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Message).where(Message.id == message_id)
            )
            message = result.scalar_one_or_none()
            if not message:
                return False

            await session.delete(message)
            await session.commit()
            return True

    async def delete_pair(self, message_id: str) -> List[str]:
        """
        Delete a message and its paired question/answer.
        Returns a list of IDs that were deleted.
        """
        session = await self._get_session()
        async with session.begin():
            # Get the target message
            result = await session.execute(
                select(Message).where(Message.id == message_id)
            )
            target = result.scalar_one_or_none()
            if not target:
                return []

            target_role = target.role
            target_thread = target.thread_id
            target_created = target.created_at

            deleted_ids = [message_id]

            # Find the candidate pair
            pair_id = None
            if target_role == "assistant":
                # Search for the user message immediately preceding it
                result = await session.execute(
                    select(Message)
                    .where(
                        Message.thread_id == target_thread,
                        Message.role == "user",
                        Message.created_at <= target_created,
                        Message.id != message_id
                    )
                    .order_by(Message.created_at.desc())
                    .limit(1)
                )
                pair = result.scalar_one_or_none()
                if pair:
                    pair_id = pair.id
            elif target_role == "user":
                # Search for the assistant message immediately following it
                result = await session.execute(
                    select(Message)
                    .where(
                        Message.thread_id == target_thread,
                        Message.role == "assistant",
                        Message.created_at >= target_created,
                        Message.id != message_id
                    )
                    .order_by(Message.created_at.asc())
                    .limit(1)
                )
                pair = result.scalar_one_or_none()
                if pair:
                    pair_id = pair.id

            if pair_id:
                deleted_ids.append(pair_id)

            # Perform the deletion
            for id_to_delete in deleted_ids:
                result = await session.execute(
                    select(Message).where(Message.id == id_to_delete)
                )
                msg = result.scalar_one_or_none()
                if msg:
                    await session.delete(msg)

            await session.commit()
            return deleted_ids

    async def delete_for_thread(self, thread_id: str) -> int:
        """Delete all messages for a thread. Returns count deleted."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Message).where(Message.thread_id == thread_id)
            )
            messages = result.scalars().all()
            count = len(messages)

            for message in messages:
                await session.delete(message)

            await session.commit()
            return count

    async def get_count(self, thread_id: str) -> int:
        """Get the total number of messages in a thread."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Message).where(Message.thread_id == thread_id)
            )
            return len(list(result.scalars().all()))
