"""
message_repo.py - Message CRUD operations.

This module provides repository methods for managing message entities in the database.
"""

import uuid
from datetime import datetime
from typing import List, Optional

import aiosqlite

from app.db.config import DB_PATH
from app.db.models import Message, MessageRole
from app.db.repositories.base import BaseRepository
from app.db.status import _parse_json_list


class MessageRepository(BaseRepository):
    """Repository for message database operations."""

    async def create(
        self,
        thread_id: str,
        role: MessageRole,
        content: str,
        context_compact: Optional[str] = None,
        reasoning: Optional[str] = None,
        reasoning_available: bool = False,
        reasoning_format: str = "none",
        web_sources: Optional[List[dict]] = None,
    ) -> Message:
        """Create a new message in a thread."""
        import json
        message_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        web_sources_json = json.dumps(web_sources) if web_sources else None

        await self._execute(
            """
            INSERT INTO messages (
                id, thread_id, role, content, context_compact, reasoning, reasoning_available, reasoning_format, web_sources, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                thread_id,
                role.value,
                content,
                context_compact,
                reasoning,
                int(reasoning_available),
                reasoning_format,
                web_sources_json,
                created_at,
            )
        )

        return Message(
            id=message_id,
            thread_id=thread_id,
            role=role,
            content=content,
            context_compact=context_compact,
            reasoning=reasoning,
            reasoning_available=reasoning_available,
            reasoning_format=reasoning_format,
            web_sources=web_sources,
            created_at=created_at
        )

    async def get(self, message_id: str) -> Optional[Message]:
        """Get a message by ID."""
        row = await self._fetch_one(
            """
            SELECT id, thread_id, role, content, reasoning, reasoning_available, reasoning_format,
                   context_compact, web_sources, created_at
            FROM messages
            WHERE id = ?
            """,
            (message_id,)
        )
        if row:
            return Message(
                id=row["id"],
                thread_id=row["thread_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                context_compact=row["context_compact"],
                reasoning=row["reasoning"],
                reasoning_available=bool(row["reasoning_available"]),
                reasoning_format=row["reasoning_format"] or "none",
                web_sources=_parse_json_list(row["web_sources"]),
                created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"]
            )
        return None

    async def get_thread_messages(
        self,
        thread_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Message]:
        """Get messages for a thread with pagination."""
        rows = await self._fetch_all("""
            SELECT id, thread_id, role, content, context_compact, reasoning, reasoning_available,
                   reasoning_format, web_sources, created_at
            FROM messages
            WHERE thread_id = ?
            ORDER BY created_at ASC
            LIMIT ? OFFSET ?
        """, (thread_id, limit, offset))
        return [
            Message(
                id=row["id"],
                thread_id=row["thread_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                context_compact=row["context_compact"],
                reasoning=row["reasoning"],
                reasoning_available=bool(row["reasoning_available"]),
                reasoning_format=row["reasoning_format"] or "none",
                web_sources=_parse_json_list(row["web_sources"]),
                created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"]
            )
            for row in rows
        ]

    async def get_recent_messages(self, thread_id: str, limit: int = 10) -> List[Message]:
        """Get the most recent messages for a thread (for context window)."""
        rows = await self._fetch_all("""
            SELECT id, thread_id, role, content, context_compact, reasoning, reasoning_available,
                   reasoning_format, web_sources, created_at
            FROM messages
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (thread_id, limit))
        # Reverse to get chronological order
        messages = [
            Message(
                id=row["id"],
                thread_id=row["thread_id"],
                role=MessageRole(row["role"]),
                content=row["content"],
                context_compact=row["context_compact"],
                reasoning=row["reasoning"],
                reasoning_available=bool(row["reasoning_available"]),
                reasoning_format=row["reasoning_format"] or "none",
                web_sources=_parse_json_list(row["web_sources"]),
                created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"]
            )
            for row in rows
        ]
        return list(reversed(messages))

    async def update_context_compact(self, message_id: str, context_compact: str) -> bool:
        """Update compact context text for a message."""
        cursor = await self._execute(
            "UPDATE messages SET context_compact = ? WHERE id = ?",
            (context_compact, message_id),
        )
        return cursor.rowcount > 0

    async def delete(self, message_id: str) -> bool:
        """Delete a message by ID."""
        cursor = await self._execute("DELETE FROM messages WHERE id = ?", (message_id,))
        return cursor.rowcount > 0

    async def delete_pair(self, message_id: str) -> List[str]:
        """
        Delete a message and its paired question/answer.
        Returns a list of IDs that were deleted.
        """
        async def _delete_pair_in_transaction(db):
            db.row_factory = aiosqlite.Row
            # 1. Get the target message
            cursor = await db.execute("SELECT id, thread_id, role, created_at FROM messages WHERE id = ?", (message_id,))
            target = await cursor.fetchone()
            if not target:
                return []

            target_role = target["role"]
            target_thread = target["thread_id"]
            target_created = target["created_at"]

            deleted_ids = [message_id]

            # 2. Find the candidate pair
            pair_id = None
            if target_role == "assistant":
                # Search for the user message immediately preceding it
                cursor = await db.execute("""
                    SELECT id FROM messages
                    WHERE thread_id = ? AND role = 'user' AND created_at <= ? AND id != ?
                    ORDER BY created_at DESC LIMIT 1
                """, (target_thread, target_created, message_id))
                pair = await cursor.fetchone()
                if pair:
                    pair_id = pair["id"]
            elif target_role == "user":
                # Search for the assistant message immediately following it
                cursor = await db.execute("""
                    SELECT id FROM messages
                    WHERE thread_id = ? AND role = 'assistant' AND created_at >= ? AND id != ?
                    ORDER BY created_at ASC LIMIT 1
                """, (target_thread, target_created, message_id))
                pair = await cursor.fetchone()
                if pair:
                    pair_id = pair["id"]

            if pair_id:
                deleted_ids.append(pair_id)

            # 3. Perform the deletion
            placeholders = ', '.join(['?'] * len(deleted_ids))
            await db.execute(f"DELETE FROM messages WHERE id IN ({placeholders})", deleted_ids)
            await db.commit()
            return deleted_ids

        return await self._transaction(_delete_pair_in_transaction)

    async def get_count(self, thread_id: str) -> int:
        """Get the total number of messages in a thread."""
        row = await self._fetch_one(
            "SELECT COUNT(*) FROM messages WHERE thread_id = ?",
            (thread_id,)
        )
        return row[0] if row else 0
