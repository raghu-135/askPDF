"""
thread_file_repo.py - Thread-file association operations.

This module provides repository methods for managing thread-file associations
and annotations in the database.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

import aiosqlite

from app.db.config import DB_PATH
from app.db.models import File, ThreadFileAnnotation
from app.db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ThreadFileRepository(BaseRepository):
    """Repository for thread-file association database operations."""

    async def add(self, thread_id: str, file_hash: str) -> bool:
        """Associate a file with a thread."""
        try:
            await self._execute(
                "INSERT OR IGNORE INTO thread_files (thread_id, file_hash) VALUES (?, ?)",
                (thread_id, file_hash)
            )
            return True
        except Exception as e:
            logger.error(f"Error adding file to thread: {e}")
            return False

    async def get_files(self, thread_id: str) -> List[File]:
        """Get all files associated with a thread."""
        rows = await self._fetch_all("""
            SELECT f.file_hash, f.file_name, f.file_path, f.source_type
            FROM files f
            JOIN thread_files tf ON f.file_hash = tf.file_hash
            WHERE tf.thread_id = ?
            ORDER BY tf.added_at DESC
        """, (thread_id,))
        return [
            File(
                file_hash=row["file_hash"],
                file_name=row["file_name"],
                file_path=row["file_path"],
                source_type=row["source_type"] or "pdf",
            )
            for row in rows
        ]

    async def remove(self, thread_id: str, file_hash: str) -> bool:
        """Remove a file association from a thread (does not delete the file record itself)."""
        try:
            async def _remove_in_transaction(db):
                await db.execute(
                    "DELETE FROM thread_file_annotations WHERE thread_id = ? AND file_hash = ?",
                    (thread_id, file_hash)
                )
                cursor = await db.execute(
                    "DELETE FROM thread_files WHERE thread_id = ? AND file_hash = ?",
                    (thread_id, file_hash)
                )
                await db.commit()
                return cursor.rowcount > 0

            return await self._transaction(_remove_in_transaction)
        except Exception as e:
            logger.error(f"Error removing file from thread: {e}")
            return False

    async def is_file_in_thread(self, thread_id: str, file_hash: str) -> bool:
        """Check if a file is associated with a thread."""
        row = await self._fetch_one(
            "SELECT 1 FROM thread_files WHERE thread_id = ? AND file_hash = ?",
            (thread_id, file_hash)
        )
        return row is not None

    async def count_threads_with_file_for_model(
        self,
        file_hash: str,
        embed_model: str,
        exclude_thread_id: Optional[str] = None,
    ) -> int:
        """
        Count thread associations for a file restricted to a specific embedding model.
        Optionally exclude one thread (useful after detaching a file from that thread).
        """
        if exclude_thread_id:
            row = await self._fetch_one(
                """
                SELECT COUNT(*)
                FROM thread_files tf
                JOIN threads t ON t.id = tf.thread_id
                WHERE tf.file_hash = ? AND t.embed_model = ? AND tf.thread_id != ?
                """,
                (file_hash, embed_model, exclude_thread_id),
            )
        else:
            row = await self._fetch_one(
                """
                SELECT COUNT(*)
                FROM thread_files tf
                JOIN threads t ON t.id = tf.thread_id
                WHERE tf.file_hash = ? AND t.embed_model = ?
                """,
                (file_hash, embed_model),
            )
        return int(row[0]) if row else 0

    async def count_threads_with_file(self, file_hash: str) -> int:
        """Count how many threads currently reference a file."""
        row = await self._fetch_one(
            "SELECT COUNT(*) FROM thread_files WHERE file_hash = ?",
            (file_hash,),
        )
        return int(row[0]) if row else 0

    def _load_annotations(self, raw: Optional[str]) -> List[Dict[str, Any]]:
        """Deserialize the annotation snapshot list from SQLite."""
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []

    def _serialize_annotation_row(self, row: ThreadFileAnnotation) -> Dict[str, Any]:
        """Convert an annotation row into the API payload shape."""
        return {
            "thread_id": row.thread_id,
            "file_hash": row.file_hash,
            "annotations": self._load_annotations(row.annotations_json),
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    async def get_annotation_row(
        self,
        thread_id: str,
        file_hash: str,
    ) -> Optional[ThreadFileAnnotation]:
        """Load the persisted row for a thread/file annotation snapshot."""
        row = await self._fetch_one(
            """
            SELECT thread_id, file_hash, annotations_json, created_at, updated_at
            FROM thread_file_annotations
            WHERE thread_id = ? AND file_hash = ?
            """,
            (thread_id, file_hash),
        )
        if not row:
            return None
        return ThreadFileAnnotation(
            thread_id=row["thread_id"],
            file_hash=row["file_hash"],
            annotations_json=row["annotations_json"] or "[]",
            created_at=datetime.fromisoformat(row["created_at"]) if isinstance(row["created_at"], str) else row["created_at"],
            updated_at=datetime.fromisoformat(row["updated_at"]) if isinstance(row["updated_at"], str) else row["updated_at"],
        )

    async def get_annotations(self, thread_id: str, file_hash: str) -> Optional[Dict[str, Any]]:
        """Get the persisted annotation payload for a thread/file pair."""
        row = await self.get_annotation_row(thread_id, file_hash)
        if not row:
            return None
        return self._serialize_annotation_row(row)

    async def upsert_annotations(
        self,
        thread_id: str,
        file_hash: str,
        annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Insert or replace the full annotation snapshot for a thread/file pair."""
        annotations_json = json.dumps(annotations or [])
        await self._execute(
            """
            INSERT INTO thread_file_annotations (
                thread_id, file_hash, annotations_json, created_at, updated_at
            ) VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(thread_id, file_hash) DO UPDATE SET
                annotations_json = excluded.annotations_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (thread_id, file_hash, annotations_json),
        )

        row = await self.get_annotation_row(thread_id, file_hash)
        if not row:
            raise RuntimeError("Failed to persist annotation snapshot")
        return self._serialize_annotation_row(row)

    async def delete_annotations(self, thread_id: str, file_hash: Optional[str] = None) -> int:
        """Delete persisted annotations for a thread or thread/file pair."""
        if file_hash:
            cursor = await self._execute(
                "DELETE FROM thread_file_annotations WHERE thread_id = ? AND file_hash = ?",
                (thread_id, file_hash),
            )
        else:
            cursor = await self._execute(
                "DELETE FROM thread_file_annotations WHERE thread_id = ?",
                (thread_id,),
            )
        return cursor.rowcount or 0
