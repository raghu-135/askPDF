"""
stats_repo_sqlmodel.py - Thread stats operations with SQLModel.

This module provides repository methods for managing denormalized thread
statistics and document metadata stored on the threads table.
"""

import json
from typing import Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from sqlalchemy.orm.attributes import flag_modified

from app.db.models_sqlmodel import Thread, ChatTurn, File, ThreadFile
from app.db.connection_sqlmodel import async_session_maker
from app.time_utils import maybe_iso_utc_z
from app.time_utils import utc_now


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

    def _replace_documents_meta(self, thread: Thread, documents_meta: Dict[str, Any]) -> None:
        """Replace thread.documents_meta without adding synthetic cache entries."""
        thread.documents_meta = dict(documents_meta)
        flag_modified(thread, "documents_meta")
        thread.stats_last_updated_at = utc_now()

    async def get_or_create(self, thread_id: str) -> Dict[str, Any]:
        """Get thread stats values, returning an empty shape for missing threads."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Thread).where(Thread.id == thread_id)
            )
            thread = result.scalar_one_or_none()

            if not thread:
                return {
                    "thread_id": thread_id,
                    "total_qa_pairs": 0,
                    "total_qa_chars": 0,
                    "avg_qa_chars": 0.0,
                    "last_qa_at": None,
                    "documents_meta": {},
                }

        return {
            "thread_id": thread.id,
            "total_qa_pairs": thread.total_qa_pairs,
            "total_qa_chars": thread.total_qa_chars,
            "avg_qa_chars": thread.avg_qa_chars,
            "last_qa_at": thread.last_qa_at,
            "documents_meta": self._load_documents_meta(thread.documents_meta),
        }

    async def record_qa(self, thread_id: str, qa_chars: int) -> None:
        """Record a QA interaction - increment counters."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Thread).where(Thread.id == thread_id)
            )
            thread = result.scalar_one_or_none()

            if not thread:
                return

            new_pairs = thread.total_qa_pairs + 1
            new_chars = thread.total_qa_chars + qa_chars
            now = utc_now()
            thread.total_qa_pairs = new_pairs
            thread.total_qa_chars = new_chars
            thread.avg_qa_chars = new_chars / new_pairs
            thread.last_qa_at = now
            thread.stats_last_updated_at = now

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
        """Insert or replace a document entry in thread.documents_meta."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Thread).where(Thread.id == thread_id)
            )
            thread = result.scalar_one_or_none()

            if not thread:
                return

            current_meta = self._load_documents_meta(thread.documents_meta)
            current = current_meta.get(file_hash, {})
            current_meta[file_hash] = {**(current if isinstance(current, dict) else {}), **meta}
            self._replace_documents_meta(thread, current_meta)

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
        """Remove a document entry from thread.documents_meta."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Thread).where(Thread.id == thread_id)
            )
            thread = result.scalar_one_or_none()

            if not thread:
                return

            current_meta = self._load_documents_meta(thread.documents_meta)
            if file_hash in current_meta:
                del current_meta[file_hash]
                self._replace_documents_meta(thread, current_meta)
                await session.flush()

    async def recompute_qa_stats(self, thread_id: str) -> None:
        """
        Recompute QA stats from chat turns.
        Called after message pair deletion to prevent drift.
        """
        session = await self._get_session()
        async with session.begin():
            # Count completed/clarification turns with visible answers.
            result = await session.execute(
                select(
                    func.count(ChatTurn.id).label("cnt"),
                    func.coalesce(
                        func.sum(func.length(ChatTurn.payload["answer"].astext)),
                        0,
                    ).label("total_chars")
                )
                .where(
                    ChatTurn.thread_id == thread_id,
                    ChatTurn.status.in_(["completed", "clarification"]),
                    ChatTurn.payload["answer"].astext.isnot(None),
                    ChatTurn.payload["answer"].astext != "",
                )
            )
            row = result.one()
            cnt = row.cnt or 0
            total_chars = row.total_chars or 0
            avg = (total_chars / cnt) if cnt > 0 else 0.0

            result = await session.execute(
                select(Thread).where(Thread.id == thread_id)
            )
            thread = result.scalar_one_or_none()

            if not thread:
                return

            thread.total_qa_pairs = cnt
            thread.total_qa_chars = total_chars
            thread.avg_qa_chars = avg
            thread.stats_last_updated_at = utc_now()

            await session.flush()

    async def get_stats(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get thread stats with computed fields."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Thread).where(Thread.id == thread_id)
            )
            thread = result.scalar_one_or_none()

            if not thread:
                return {
                    "total_qa_pairs": 0,
                    "total_qa_chars": 0,
                    "avg_qa_chars": 0.0,
                    "last_qa_at": None,
                    "documents": {},
                }

            return {
                "total_qa_pairs": thread.total_qa_pairs,
                "total_qa_chars": thread.total_qa_chars,
                "avg_qa_chars": round(thread.avg_qa_chars, 1),
                "last_qa_at": thread.last_qa_at,
                "documents": self._load_documents_meta(thread.documents_meta),
            }

    async def get_thread_shape(self, thread_id: str) -> Dict[str, Any]:
        """
        Return a structured snapshot of the thread's content inventory.
        Used by the prefetch path in chat_service and by the get_thread_shape agent tool.

        Thread-file associations are the source of truth for document membership.
        The thread.documents_meta JSON is only a cache for indexing details.
        """
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Thread).where(Thread.id == thread_id)
            )
            thread = result.scalar_one_or_none()

            files_result = await session.execute(
                select(File, ThreadFile.added_at)
                .join(ThreadFile, File.file_hash == ThreadFile.file_hash)
                .where(ThreadFile.thread_id == thread_id)
                .order_by(ThreadFile.added_at.asc())
            )
            attached_files = list(files_result.all())

        if thread:
            documents_cache = self._load_documents_meta(thread.documents_meta)
            total_qa_pairs = thread.total_qa_pairs
            total_qa_chars = thread.total_qa_chars
            avg_qa_chars = round(thread.avg_qa_chars, 1)
            last_qa_at = thread.last_qa_at
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
                "document_available_in_thread_at": maybe_iso_utc_z(added_at),
            }

        return {
            "total_qa_pairs": total_qa_pairs,
            "total_qa_chars": total_qa_chars,
            "avg_qa_chars": avg_qa_chars,
            "last_qa_at": last_qa_at,
            "documents": documents,
        }
