"""
thread_repo.py - Thread CRUD operations.

This module provides repository methods for managing thread entities in the database.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import aiosqlite

from app.db.config import DB_PATH
from app.db.models import Thread
from app.db.repositories.base import BaseRepository
from app.db.status import _parse_settings


class ThreadRepository(BaseRepository):
    """Repository for thread database operations."""

    async def create(self, name: str, embed_model: str) -> Thread:
        """Create a new thread."""
        thread_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute(
                "INSERT INTO threads (id, name, embed_model, settings, created_at) VALUES (?, ?, ?, ?, ?)",
                (thread_id, name, embed_model, "{}", created_at)
            )
            await db.commit()

        return Thread(id=thread_id, name=name, embed_model=embed_model, settings={}, created_at=created_at)

    async def get(self, thread_id: str) -> Optional[Thread]:
        """Get a thread by ID."""
        row = await self._fetch_one(
            "SELECT id, name, embed_model, settings, created_at FROM threads WHERE id = ?",
            (thread_id,)
        )
        if row:
            return Thread(
                id=row["id"],
                name=row["name"],
                embed_model=row["embed_model"],
                settings=_parse_settings(row["settings"]),
                created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"]
            )
        return None

    async def get_settings(self, thread_id: str) -> Dict[str, Any]:
        """Get persisted settings for a thread."""
        row = await self._fetch_one(
            "SELECT settings FROM threads WHERE id = ?",
            (thread_id,)
        )
        if not row:
            return {}
        return _parse_settings(row["settings"])

    async def update_settings(self, thread_id: str, settings: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Replace persisted settings for a thread."""
        import json
        payload = json.dumps(settings or {})
        cursor = await self._execute(
            "UPDATE threads SET settings = ? WHERE id = ?",
            (payload, thread_id)
        )
        if cursor.rowcount == 0:
            return None
        return await self.get_settings(thread_id)

    async def list_all(self) -> List[Dict[str, Any]]:
        """List all threads with message counts and file counts."""
        rows = await self._fetch_all("""
            SELECT
                t.id, t.name, t.embed_model, t.settings, t.created_at,
                COUNT(DISTINCT m.id) as message_count,
                COUNT(DISTINCT tf.file_hash) as file_count,
                MAX(m.created_at) as last_message_at
            FROM threads t
            LEFT JOIN messages m ON t.id = m.thread_id
            LEFT JOIN thread_files tf ON t.id = tf.thread_id
            GROUP BY t.id
            ORDER BY COALESCE(last_message_at, t.created_at) DESC
        """)
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "embed_model": row["embed_model"],
                "settings": _parse_settings(row["settings"]),
                "created_at": row["created_at"],
                "message_count": row["message_count"],
                "file_count": row["file_count"]
            }
            for row in rows
        ]

    async def update(self, thread_id: str, name: str) -> Optional[Thread]:
        """Update a thread's name."""
        await self._execute(
            "UPDATE threads SET name = ? WHERE id = ?",
            (name, thread_id)
        )
        return await self.get(thread_id)

    async def delete(self, thread_id: str) -> bool:
        """Delete a thread and all associated data."""
        async def _delete_in_transaction(db):
            # Delete messages first (cascade should handle this, but being explicit)
            await db.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            await db.execute("DELETE FROM thread_file_annotations WHERE thread_id = ?", (thread_id,))
            # Delete thread_files associations
            await db.execute("DELETE FROM thread_files WHERE thread_id = ?", (thread_id,))
            # Delete the thread
            cursor = await db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            return cursor.rowcount > 0

        return await self._transaction(_delete_in_transaction)
