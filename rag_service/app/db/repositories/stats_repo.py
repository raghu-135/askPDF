"""
stats_repo.py - Thread stats operations.

This module provides repository methods for managing thread statistics
and document metadata in the database.
"""

import json
from typing import Dict, Any, Optional

import aiosqlite

from app.db.config import DB_PATH
from app.db.repositories.base import BaseRepository


class StatsRepository(BaseRepository):
    """Repository for thread stats database operations."""

    def _load_documents_meta(self, raw: Optional[str]) -> Dict[str, Any]:
        """Deserialize the documents_meta JSON column."""
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    async def _ensure_thread_stats_row(self, db, thread_id: str) -> None:
        """Insert a thread_stats row if one doesn't exist yet."""
        await db.execute(
            "INSERT OR IGNORE INTO thread_stats (thread_id) VALUES (?)",
            (thread_id,),
        )

    async def remove_document_from_stats(self, thread_id: str, file_hash: str) -> None:
        """Remove a document entry from thread_stats.documents_meta."""
        async def _remove_in_transaction(db):
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT documents_meta FROM thread_stats WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return
            docs = self._load_documents_meta(row["documents_meta"])
            docs.pop(file_hash, None)
            await db.execute(
                """
                UPDATE thread_stats
                SET documents_meta = ?, last_updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = ?
                """,
                (json.dumps(docs), thread_id),
            )
            await db.commit()

        await self._transaction(_remove_in_transaction)

    async def upsert_document_in_stats(self, thread_id: str, file_hash: str, meta: Dict[str, Any]) -> None:
        """Insert or replace a document entry in thread_stats.documents_meta."""
        async def _upsert_in_transaction(db):
            db.row_factory = aiosqlite.Row
            await self._ensure_thread_stats_row(db, thread_id)
            cursor = await db.execute(
                "SELECT documents_meta FROM thread_stats WHERE thread_id = ?",
                (thread_id,),
            )
            row = await cursor.fetchone()
            docs = self._load_documents_meta(row["documents_meta"] if row else None)
            docs[file_hash] = {
                **docs.get(file_hash, {}),
                **(meta or {}),
            }
            await db.execute(
                """
                UPDATE thread_stats
                SET documents_meta = ?, last_updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = ?
                """,
                (json.dumps(docs), thread_id),
            )
            await db.commit()

        await self._transaction(_upsert_in_transaction)

    async def increment_qa_stats(self, thread_id: str, qa_chars: int) -> None:
        """
        Increment QA aggregate counters after each answered turn.
        Called on the hot path (every chat answer).
        """
        async def _increment_in_transaction(db):
            await self._ensure_thread_stats_row(db, thread_id)
            await db.execute(
                """
                UPDATE thread_stats
                SET
                    total_qa_pairs  = total_qa_pairs + 1,
                    total_qa_chars  = total_qa_chars + ?,
                    avg_qa_chars    = CAST(total_qa_chars + ? AS REAL) / (total_qa_pairs + 1),
                    last_qa_at      = CURRENT_TIMESTAMP,
                    last_updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = ?
                """,
                (qa_chars, qa_chars, thread_id),
            )
            await db.commit()

        await self._transaction(_increment_in_transaction)

    async def recompute_qa_stats(self, thread_id: str) -> None:
        """
        Recompute QA stats from the messages table.
        Called after message pair deletion (rare path) to prevent drift.
        Counts assistant messages as QA pairs and sums their content chars.
        """
        async def _recompute_in_transaction(db):
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT COUNT(*) as cnt, COALESCE(SUM(LENGTH(content)), 0) as total_chars
                FROM messages
                WHERE thread_id = ? AND role = 'assistant'
                """,
                (thread_id,),
            )
            row = await cursor.fetchone()
            cnt = row["cnt"] or 0
            total_chars = row["total_chars"] or 0
            avg = (total_chars / cnt) if cnt > 0 else 0.0

            await self._ensure_thread_stats_row(db, thread_id)
            await db.execute(
                """
                UPDATE thread_stats
                SET total_qa_pairs  = ?,
                    total_qa_chars  = ?,
                    avg_qa_chars    = ?,
                    last_updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = ?
                """,
                (cnt, total_chars, avg, thread_id),
            )
            await db.commit()

        await self._transaction(_recompute_in_transaction)

    async def get_thread_shape(self, thread_id: str) -> Dict[str, Any]:
        """
        Return a structured snapshot of the thread's content inventory.
        Used by the prefetch path in chat_service and by the get_thread_shape agent tool.

        Returns:
            {
              "total_qa_pairs": int,
              "total_qa_chars": int,
              "avg_qa_chars": float,
              "last_qa_at": str | None,
              "documents": {
                  "<file_hash>": {
                      "file_name": str,
                      "source_type": str,
                      "chunk_count": int,
                      "total_chars": int,
                      "indexing_status": str,
                      "indexed_at": str | None
                  }
              }
        }
        """
        row = await self._fetch_one(
            """
            SELECT total_qa_pairs, total_qa_chars, avg_qa_chars,
                   last_qa_at, documents_meta
            FROM thread_stats WHERE thread_id = ?
            """,
            (thread_id,),
        )

        if not row:
            return {
                "total_qa_pairs": 0,
                "total_qa_chars": 0,
                "avg_qa_chars": 0.0,
                "last_qa_at": None,
                "documents": {},
            }

        return {
            "total_qa_pairs": row["total_qa_pairs"],
            "total_qa_chars": row["total_qa_chars"],
            "avg_qa_chars": round(row["avg_qa_chars"], 1),
            "last_qa_at": row["last_qa_at"],
            "documents": self._load_documents_meta(row["documents_meta"]),
        }
