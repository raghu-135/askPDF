"""
stats_repo_sqlmodel.py - Thread stats operations with SQLModel.

This module provides repository methods for managing thread statistics
and document metadata using SQLModel with PostgreSQL, including JSONB
documents_meta handling.
"""

import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from app.db.models_sqlmodel import ThreadStats, Thread, Message, File, ThreadFile
from app.db.jsonb_utils import merge_jsonb_field
from app.db.connection_sqlmodel import async_session_maker


def _iso_utc(value: Any) -> Any:
    if not value:
        return value
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            raw = str(value)
            if raw.endswith("Z"):
                return raw
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


class StatsRepository:
    """Repository for thread stats database operations using SQLModel."""

    def __init__(self, session: Optional[AsyncSession] = None):
        """Initialize with optional session for dependency injection."""
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get a database session - injected for tests, default for production."""
        if self._session is not None:
            return self._session
        return async_session_maker()

    def _load_documents_meta(self, raw: Optional[Any]) -> Dict[str, Any]:
        """Deserialize the documents_meta JSON column."""
        if not raw:
            return {}
        # If already a dict, return it directly
        if isinstance(raw, dict):
            return raw
        # If it's a string, try to parse it as JSON
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}
        return {}

    async def get_or_create(self, thread_id: str) -> Dict[str, Any]:
        """Get or create thread stats row."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadStats).where(ThreadStats.thread_id == thread_id)
            )
            stats = result.scalar_one_or_none()

            if not stats:
                stats = ThreadStats(
                    thread_id=thread_id,
                    total_qa_pairs=0,
                    total_qa_chars=0,
                    avg_qa_chars=0.0,
                    documents_meta={},
                )
                session.add(stats)
                await session.flush()
                await session.refresh(stats)

        return {
                "thread_id": stats.thread_id,
                "total_qa_pairs": stats.total_qa_pairs,
                "total_qa_chars": stats.total_qa_chars,
                "avg_qa_chars": stats.avg_qa_chars,
                "last_qa_at": stats.last_qa_at,
                "documents_meta": stats.documents_meta if stats.documents_meta else {},
            }

    async def record_qa(self, thread_id: str, qa_chars: int) -> None:
        """Record a QA interaction - increment counters."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadStats).where(ThreadStats.thread_id == thread_id)
            )
            stats = result.scalar_one_or_none()

            if not stats:
                # Create new stats row
                stats = ThreadStats(
                    thread_id=thread_id,
                    total_qa_pairs=1,
                    total_qa_chars=qa_chars,
                    avg_qa_chars=float(qa_chars),
                    documents_meta={},
                )
                session.add(stats)
            else:
                # Update existing
                new_pairs = stats.total_qa_pairs + 1
                new_chars = stats.total_qa_chars + qa_chars
                stats.total_qa_pairs = new_pairs
                stats.total_qa_chars = new_chars
                stats.avg_qa_chars = new_chars / new_pairs

            await session.flush()

    async def increment_qa_stats(self, thread_id: str, qa_chars: int) -> None:
        """Alias for record_qa for backward compatibility."""
        await self.record_qa(thread_id, qa_chars)

    async def update_documents_meta(
        self,
        thread_id: str,
        file_hash: str,
        meta: Dict[str, Any]
    ) -> None:
        """Insert or replace a document entry in thread_stats.documents_meta."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadStats).where(ThreadStats.thread_id == thread_id)
            )
            stats = result.scalar_one_or_none()

            if not stats:
                # Create new stats row with document meta
                stats = ThreadStats(
                    thread_id=thread_id,
                    total_qa_pairs=0,
                    total_qa_chars=0,
                    avg_qa_chars=0.0,
                    documents_meta={file_hash: meta},
                )
                session.add(stats)
            else:
                # Update existing using JSONB merge
                current_meta = dict(stats.documents_meta or {})
                current_meta[file_hash] = {**current_meta.get(file_hash, {}), **meta}
                merge_jsonb_field(stats, "documents_meta", current_meta)

            await session.flush()

    async def upsert_document_in_stats(
        self,
        thread_id: str,
        file_hash: str,
        meta: Dict[str, Any]
    ) -> None:
        """Alias for update_documents_meta for backward compatibility."""
        await self.update_documents_meta(thread_id, file_hash, meta)

    async def remove_document_from_stats(
        self,
        thread_id: str,
        file_hash: str
    ) -> None:
        """Remove a document entry from thread_stats.documents_meta."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadStats).where(ThreadStats.thread_id == thread_id)
            )
            stats = result.scalar_one_or_none()

            if not stats:
                return

            current_meta = dict(stats.documents_meta or {})
            if file_hash in current_meta:
                del current_meta[file_hash]
                merge_jsonb_field(stats, "documents_meta", current_meta)
                await session.flush()

    async def recompute_qa_stats(self, thread_id: str) -> None:
        """
        Recompute QA stats from the messages table.
        Called after message pair deletion to prevent drift.
        """
        session = await self._get_session()
        async with session.begin():
            # Count assistant messages and sum their content chars
            result = await session.execute(
                select(
                    func.count(Message.id).label("cnt"),
                    func.coalesce(func.sum(func.length(Message.content)), 0).label("total_chars")
                )
                .where(Message.thread_id == thread_id, Message.role == "assistant")
            )
            row = result.one()
            cnt = row.cnt or 0
            total_chars = row.total_chars or 0
            avg = (total_chars / cnt) if cnt > 0 else 0.0

            # Get or create stats row
            result = await session.execute(
                select(ThreadStats).where(ThreadStats.thread_id == thread_id)
            )
            stats = result.scalar_one_or_none()

            if not stats:
                stats = ThreadStats(
                    thread_id=thread_id,
                    total_qa_pairs=cnt,
                    total_qa_chars=total_chars,
                    avg_qa_chars=avg,
                    documents_meta={},
                )
                session.add(stats)
            else:
                stats.total_qa_pairs = cnt
                stats.total_qa_chars = total_chars
                stats.avg_qa_chars = avg

            await session.flush()

    async def get_stats(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get thread stats with computed fields."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadStats).where(ThreadStats.thread_id == thread_id)
            )
            stats = result.scalar_one_or_none()

            if not stats:
                return {
                    "total_qa_pairs": 0,
                    "total_qa_chars": 0,
                    "avg_qa_chars": 0.0,
                    "last_qa_at": None,
                    "documents": {},
                }

            return {
                "total_qa_pairs": stats.total_qa_pairs,
                "total_qa_chars": stats.total_qa_chars,
                "avg_qa_chars": round(stats.avg_qa_chars, 1),
                "last_qa_at": stats.last_qa_at,
                "documents": self._load_documents_meta(stats.documents_meta),
            }

    async def get_thread_shape(self, thread_id: str) -> Dict[str, Any]:
        """
        Return a structured snapshot of the thread's content inventory.
        Used by the prefetch path in chat_service and by the get_thread_shape agent tool.

        Thread-file associations are the source of truth for document membership.
        The thread_stats.documents_meta JSON is only a cache for indexing details.
        """
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadStats).where(ThreadStats.thread_id == thread_id)
            )
            stats_row = result.scalar_one_or_none()

            files_result = await session.execute(
                select(File, ThreadFile.added_at)
                .join(ThreadFile, File.file_hash == ThreadFile.file_hash)
                .where(ThreadFile.thread_id == thread_id)
                .order_by(ThreadFile.added_at.asc())
            )
            attached_files = list(files_result.all())

        if stats_row:
            documents_cache = self._load_documents_meta(stats_row.documents_meta)
            total_qa_pairs = stats_row.total_qa_pairs
            total_qa_chars = stats_row.total_qa_chars
            avg_qa_chars = round(stats_row.avg_qa_chars, 1)
            last_qa_at = stats_row.last_qa_at
        else:
            documents_cache = {}
            total_qa_pairs = 0
            total_qa_chars = 0
            avg_qa_chars = 0.0
            last_qa_at = None

        documents: Dict[str, Any] = {}
        for file, added_at in attached_files:
            cached = documents_cache.get(file.file_hash, {})
            cached = cached if isinstance(cached, dict) else {}
            documents[file.file_hash] = {
                **cached,
                "file_name": file.file_name,
                "file_hash": file.file_hash,
                "source_type": file.source_type,
                "file_path": file.file_path,
                "document_available_in_thread_at": _iso_utc(added_at),
            }

        return {
            "total_qa_pairs": total_qa_pairs,
            "total_qa_chars": total_qa_chars,
            "avg_qa_chars": avg_qa_chars,
            "last_qa_at": last_qa_at,
            "documents": documents,
        }
