"""Project CRUD operations with SQLModel."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import Project, Thread
from app.time_utils import utc_now
from app.models.llm_server_client import LOCAL_EMBEDDING_MODEL
from app.services.memory_policy import merge_project_settings_json


DEFAULT_PROJECT_NAME = "Personal"


class ProjectRepository:
    """Repository for project containers."""

    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def _get_session(self) -> AsyncSession:
        if self._session is not None:
            return self._session
        return async_session_maker()

    async def ensure_default_project(self) -> Project:
        """Return the default project, creating it and backfilling orphan threads."""

        session = await self._get_session()
        async with session.begin():
            result = await session.execute(select(Project).where(Project.name == DEFAULT_PROJECT_NAME))
            project = result.scalar_one_or_none()
            if project is None:
                project = Project(
                    id=str(uuid.uuid4()),
                    name=DEFAULT_PROJECT_NAME,
                    description="Default project for existing threads.",
                    embedding_model=LOCAL_EMBEDDING_MODEL,
                    settings_json={},
                    created_at=utc_now(),
                    last_activity_at=utc_now(),
                )
                session.add(project)
                await session.flush()

            orphan_result = await session.execute(select(Thread).where(Thread.project_id.is_(None)))
            for thread in orphan_result.scalars().all():
                thread.project_id = project.id
                thread.embedding_model = project.embedding_model
            await session.flush()
            await session.refresh(project)
            return project

    async def create(
        self,
        *,
        name: str,
        embedding_model: str,
        description: str = "",
        settings_json: Optional[Dict[str, Any]] = None,
    ) -> Project:
        created_at = utc_now()
        project = Project(
            id=str(uuid.uuid4()),
            name=name,
            embedding_model=embedding_model,
            description=description or "",
            settings_json=merge_project_settings_json({}, settings_json),
            created_at=created_at,
            last_activity_at=created_at,
        )
        session = await self._get_session()
        async with session.begin():
            session.add(project)
            await session.flush()
            await session.refresh(project)
            return project

    async def get(self, project_id: str) -> Optional[Project]:
        session = await self._get_session()
        async with session.begin():
            return await session.get(Project, project_id)

    async def list_all(self) -> list[Project]:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(Project).order_by(
                    Project.last_activity_at.desc(),
                    Project.created_at.desc(),
                    Project.name.asc(),
                )
            )
            return list(result.scalars().all())

    async def update(
        self,
        project_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        settings_json: Optional[Dict[str, Any]] = None,
    ) -> Optional[Project]:
        session = await self._get_session()
        async with session.begin():
            project = await session.get(Project, project_id)
            if project is None:
                return None
            if name is not None:
                project.name = name
            if description is not None:
                project.description = description
            if settings_json is not None:
                replace_jsonb_field(
                    project,
                    "settings_json",
                    merge_project_settings_json(project.settings_json, settings_json),
                )
            await session.flush()
            await session.refresh(project)
            return project

    async def assign_thread(self, thread_id: str, project_id: str) -> Optional[Thread]:
        session = await self._get_session()
        async with session.begin():
            project = await session.get(Project, project_id)
            thread = await session.get(Thread, thread_id)
            if project is None or thread is None:
                return None
            if thread.embedding_model != project.embedding_model:
                raise ValueError(
                    "Thread cannot move to a project with a different embedding model"
                )
            thread.project_id = project.id
            await session.flush()
            await session.refresh(thread)
            return thread
