"""
thread_file_repo_sqlmodel.py - Thread-file association operations with SQLModel.

This module provides repository methods for managing thread-file associations
and annotations using SQLModel with PostgreSQL.
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.models_sqlmodel import File, Thread, ThreadFile, ThreadFileAnnotation
from app.db.connection_sqlmodel import async_session_maker

logger = logging.getLogger(__name__)


class ThreadFileRepository:
    """Repository for thread-file association database operations using SQLModel."""

    def __init__(self, session: Optional[AsyncSession] = None):
        """Initialize with optional session for dependency injection."""
        self._session = session

    async def _get_session(self) -> AsyncSession:
        """Get a database session - injected for tests, default for production."""
        if self._session is not None:
            return self._session
        return async_session_maker()

    async def add(self, thread_id: str, file_hash: str) -> bool:
        """Associate a file with a thread."""
        session = await self._get_session()
        async with session.begin():
            # Check if already exists
            result = await session.execute(
                select(ThreadFile).where(
                    ThreadFile.thread_id == thread_id,
                    ThreadFile.file_hash == file_hash
                )
            )
            existing = result.scalar_one_or_none()
            if existing:
                return True

            # Create new association
            association = ThreadFile(
                thread_id=thread_id,
                file_hash=file_hash,
                added_at=datetime.utcnow()
            )
            session.add(association)
            await session.flush()
            await session.commit()
            return True

    async def get_files(self, thread_id: str) -> List[File]:
        """Get all files associated with a thread."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(File)
                .join(ThreadFile, File.file_hash == ThreadFile.file_hash)
                .where(ThreadFile.thread_id == thread_id)
                .order_by(ThreadFile.added_at.desc())
            )
            return list(result.scalars().all())

    async def remove(self, thread_id: str, file_hash: str) -> bool:
        """Remove a file association from a thread (does not delete the file record itself)."""
        session = await self._get_session()
        async with session.begin():
            # First delete any annotations for this association
            result = await session.execute(
                select(ThreadFileAnnotation).where(
                    ThreadFileAnnotation.thread_id == thread_id,
                    ThreadFileAnnotation.file_hash == file_hash
                )
            )
            annotation = result.scalar_one_or_none()
            if annotation:
                await session.delete(annotation)

            # Then delete the association
            result = await session.execute(
                select(ThreadFile).where(
                    ThreadFile.thread_id == thread_id,
                    ThreadFile.file_hash == file_hash
                )
            )
            association = result.scalar_one_or_none()
            if not association:
                return False

            await session.delete(association)
            await session.commit()
            return True

    async def is_file_in_thread(self, thread_id: str, file_hash: str) -> bool:
        """Check if a file is associated with a thread."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadFile).where(
                    ThreadFile.thread_id == thread_id,
                    ThreadFile.file_hash == file_hash
                )
            )
            return result.scalar_one_or_none() is not None

    async def get_count_for_file(self, file_hash: str) -> int:
        """Count how many threads currently reference a file."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadFile).where(ThreadFile.file_hash == file_hash)
            )
            return len(list(result.scalars().all()))

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
        session = await self._get_session()
        async with session.begin():
            query = (
                select(ThreadFile)
                .join(Thread, ThreadFile.thread_id == Thread.id)
                .where(
                    ThreadFile.file_hash == file_hash,
                    Thread.embed_model == embed_model
                )
            )
            if exclude_thread_id:
                query = query.where(ThreadFile.thread_id != exclude_thread_id)

            result = await session.execute(query)
            return len(list(result.scalars().all()))

    async def get_all_for_thread(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get all thread-file associations for a thread."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadFile).where(ThreadFile.thread_id == thread_id)
            )
            associations = result.scalars().all()
            return [
                {
                    "thread_id": assoc.thread_id,
                    "file_hash": assoc.file_hash,
                    "added_at": assoc.added_at,
                }
                for assoc in associations
            ]

    def _load_annotations(self, raw: Optional[str]) -> List[Dict[str, Any]]:
        """Deserialize the annotation snapshot list from JSON."""
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
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadFileAnnotation).where(
                    ThreadFileAnnotation.thread_id == thread_id,
                    ThreadFileAnnotation.file_hash == file_hash
                )
            )
            return result.scalar_one_or_none()

    async def get_annotations(
        self,
        thread_id: str,
        file_hash: str
    ) -> Optional[Dict[str, Any]]:
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
        now = datetime.utcnow()

        session = await self._get_session()
        async with session.begin():
            # Try to get existing
            result = await session.execute(
                select(ThreadFileAnnotation).where(
                    ThreadFileAnnotation.thread_id == thread_id,
                    ThreadFileAnnotation.file_hash == file_hash
                )
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing
                existing.annotations_json = annotations_json
                existing.updated_at = now
                await session.flush()
                await session.commit()
                await session.refresh(existing)
                return self._serialize_annotation_row(existing)
            else:
                # Create new
                new_annotation = ThreadFileAnnotation(
                    thread_id=thread_id,
                    file_hash=file_hash,
                    annotations_json=annotations_json,
                    created_at=now,
                    updated_at=now,
                )
                session.add(new_annotation)
                await session.flush()
                await session.commit()
                await session.refresh(new_annotation)
                return self._serialize_annotation_row(new_annotation)

    async def delete_annotations(
        self,
        thread_id: str,
        file_hash: Optional[str] = None
    ) -> int:
        """Delete persisted annotations for a thread or thread/file pair."""
        session = await self._get_session()
        async with session.begin():
            if file_hash:
                result = await session.execute(
                    select(ThreadFileAnnotation).where(
                        ThreadFileAnnotation.thread_id == thread_id,
                        ThreadFileAnnotation.file_hash == file_hash
                    )
                )
                annotation = result.scalar_one_or_none()
                if annotation:
                    await session.delete(annotation)
                    await session.commit()
                    return 1
                return 0
            else:
                result = await session.execute(
                    select(ThreadFileAnnotation).where(
                        ThreadFileAnnotation.thread_id == thread_id
                    )
                )
                annotations = result.scalars().all()
                count = len(annotations)
                for annotation in annotations:
                    await session.delete(annotation)
                await session.commit()
                return count
