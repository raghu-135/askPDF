"""Project-file association and effective thread-file queries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.models_sqlmodel import File, Project, ProjectFile, Thread, ThreadFile
from app.db.project_activity import touch_project_activity
from app.time_utils import utc_now


@dataclass(frozen=True)
class EffectiveFile:
    file: File
    association_scope: str
    is_project_knowledge: bool
    added_at: datetime

    def __getattr__(self, name):
        return getattr(self.file, name)


class ProjectFileRepository:
    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def _get_session(self) -> AsyncSession:
        return self._session if self._session is not None else async_session_maker()

    async def add(self, project_id: str, file_hash: str) -> bool:
        session = await self._get_session()
        async with session.begin():
            existing = await session.get(ProjectFile, (project_id, file_hash))
            if existing:
                return True
            added_at = utc_now()
            session.add(ProjectFile(project_id=project_id, file_hash=file_hash, added_at=added_at))
            await session.flush()
            await touch_project_activity(session, project_id, occurred_at=added_at)
        return True

    async def remove(self, project_id: str, file_hash: str) -> bool:
        session = await self._get_session()
        async with session.begin():
            association = await session.get(ProjectFile, (project_id, file_hash))
            if not association:
                return False
            await session.delete(association)
            await touch_project_activity(session, project_id)
        return True

    async def is_file_in_project(self, project_id: str, file_hash: str) -> bool:
        session = await self._get_session()
        async with session.begin():
            return await session.get(ProjectFile, (project_id, file_hash)) is not None

    async def get_files(self, project_id: str) -> list[EffectiveFile]:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(File, ProjectFile.added_at)
                .join(ProjectFile, File.file_hash == ProjectFile.file_hash)
                .where(ProjectFile.project_id == project_id)
                .order_by(ProjectFile.added_at.asc())
            )
            return [
                EffectiveFile(file=row[0], association_scope="project", is_project_knowledge=True, added_at=row[1])
                for row in result.all()
            ]

    async def get_effective_thread_files(self, thread_id: str) -> list[EffectiveFile]:
        """Return direct plus inherited files, with direct associations winning."""
        session = await self._get_session()
        async with session.begin():
            thread = await session.get(Thread, thread_id)
            if not thread:
                return []
            direct_result = await session.execute(
                select(File, ThreadFile.added_at)
                .join(ThreadFile, File.file_hash == ThreadFile.file_hash)
                .where(ThreadFile.thread_id == thread_id)
                .order_by(ThreadFile.added_at.asc())
            )
            project_result = await session.execute(
                select(File, ProjectFile.added_at)
                .join(ProjectFile, File.file_hash == ProjectFile.file_hash)
                .where(ProjectFile.project_id == thread.project_id)
                .order_by(ProjectFile.added_at.asc())
            )
            project_rows = {row[0].file_hash: row for row in project_result.all()}
            effective: list[EffectiveFile] = []
            direct_hashes: set[str] = set()
            for file, added_at in direct_result.all():
                direct_hashes.add(file.file_hash)
                effective.append(EffectiveFile(
                    file=file,
                    association_scope="thread",
                    is_project_knowledge=file.file_hash in project_rows,
                    added_at=added_at,
                ))
            for file_hash, (file, added_at) in project_rows.items():
                if file_hash not in direct_hashes:
                    effective.append(EffectiveFile(
                        file=file,
                        association_scope="project",
                        is_project_knowledge=True,
                        added_at=added_at,
                    ))
            return effective

    async def is_file_accessible_to_thread(self, thread_id: str, file_hash: str) -> bool:
        session = await self._get_session()
        async with session.begin():
            direct = await session.get(ThreadFile, (thread_id, file_hash))
            if direct:
                return True
            result = await session.execute(
                select(ProjectFile.project_id)
                .join(Thread, Thread.project_id == ProjectFile.project_id)
                .where(Thread.id == thread_id, ProjectFile.file_hash == file_hash)
            )
            return result.scalar_one_or_none() is not None

    async def is_file_in_project_thread(self, project_id: str, file_hash: str) -> bool:
        """Check whether a direct attachment belongs to any thread in a project."""
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ThreadFile.thread_id)
                .join(Thread, Thread.id == ThreadFile.thread_id)
                .where(Thread.project_id == project_id, ThreadFile.file_hash == file_hash)
                .limit(1)
            )
            return result.scalar_one_or_none() is not None

    async def count_projects_with_file(self, file_hash: str) -> int:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(func.count(ProjectFile.project_id)).where(ProjectFile.file_hash == file_hash)
            )
            return int(result.scalar() or 0)

    async def count_projects_with_file_for_model(self, file_hash: str, embedding_model: str) -> int:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(func.count(ProjectFile.project_id))
                .join(Project, Project.id == ProjectFile.project_id)
                .where(ProjectFile.file_hash == file_hash, Project.embedding_model == embedding_model)
            )
            return int(result.scalar() or 0)
