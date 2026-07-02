from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.agent_patterns.templates import SUPPORTED_BUILTIN_TEMPLATE_IDS, builtin_templates
from app.agent_patterns.validator import TemplateValidator
from app.db.connection_sqlmodel import async_session_maker
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import (
    AgentPatternTemplate,
    AgentPatternTemplateVersion,
    AgentRun,
)
from app.time_utils import utc_now


class AgentPatternRepository:
    """Persistence for agent templates, template versions, and runs."""

    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def _get_session(self) -> AsyncSession:
        if self._session is not None:
            return self._session
        return async_session_maker()

    async def seed_builtin_templates(self) -> None:
        validator = TemplateValidator()
        session = await self._get_session()
        async with session.begin():
            for template_def in builtin_templates():
                version_def = template_def["version"]
                validation_result = validator.validate(version_def["spec_json"])

                template = await session.get(AgentPatternTemplate, template_def["id"])
                if template is None:
                    template = AgentPatternTemplate(
                        id=template_def["id"],
                        name=template_def["name"],
                        description=template_def["description"],
                        visibility=template_def["visibility"],
                        is_builtin=template_def["is_builtin"],
                        current_version_id=template_def["current_version_id"],
                    )
                    session.add(template)
                else:
                    template.name = template_def["name"]
                    template.description = template_def["description"]
                    template.visibility = template_def["visibility"]
                    template.is_builtin = template_def["is_builtin"]
                    template.current_version_id = template_def["current_version_id"]
                    template.updated_at = utc_now()

                version = await session.get(AgentPatternTemplateVersion, version_def["id"])
                if version is None:
                    version = AgentPatternTemplateVersion(
                        id=version_def["id"],
                        template_id=template_def["id"],
                        version=version_def["version"],
                        schema_version=version_def["schema_version"],
                        spec_json=version_def["spec_json"],
                        validation_result_json=validation_result,
                        changelog=version_def["changelog"],
                    )
                    session.add(version)
                else:
                    # Built-in v1 specs are code-owned; keep seeding idempotent and corrective.
                    version.schema_version = version_def["schema_version"]
                    replace_jsonb_field(version, "spec_json", version_def["spec_json"])
                    replace_jsonb_field(version, "validation_result_json", validation_result)
                    version.changelog = version_def["changelog"]

    async def list_templates(self) -> list[AgentPatternTemplate]:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(AgentPatternTemplate)
                .where(AgentPatternTemplate.id.in_(SUPPORTED_BUILTIN_TEMPLATE_IDS))
                .order_by(AgentPatternTemplate.name.asc())
            )
            return list(result.scalars().all())

    async def get_template(self, template_id: str) -> Optional[AgentPatternTemplate]:
        session = await self._get_session()
        async with session.begin():
            if template_id not in SUPPORTED_BUILTIN_TEMPLATE_IDS:
                return None
            return await session.get(AgentPatternTemplate, template_id)

    async def get_template_with_current_version(
        self,
        template_id: str,
    ) -> tuple[Optional[AgentPatternTemplate], Optional[AgentPatternTemplateVersion]]:
        session = await self._get_session()
        async with session.begin():
            if template_id not in SUPPORTED_BUILTIN_TEMPLATE_IDS:
                return None, None
            template = await session.get(AgentPatternTemplate, template_id)
            if not template:
                return None, None
            version = None
            if template.current_version_id:
                version = await session.get(AgentPatternTemplateVersion, template.current_version_id)
            if version is None:
                result = await session.execute(
                    select(AgentPatternTemplateVersion)
                    .where(AgentPatternTemplateVersion.template_id == template_id)
                    .order_by(AgentPatternTemplateVersion.version.desc())
                    .limit(1)
                )
                version = result.scalar_one_or_none()
            return template, version

    async def get_run(self, run_id: str) -> Optional[AgentRun]:
        session = await self._get_session()
        async with session.begin():
            return await session.get(AgentRun, run_id)

    async def create_run(
        self,
        *,
        thread_id: str,
        template_id: str,
        template_version_id: str,
        resolved_spec_json: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> AgentRun:
        run = AgentRun(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            user_id=user_id,
            template_id=template_id,
            template_version_id=template_version_id,
            resolved_spec_json=resolved_spec_json,
            status="running",
            started_at=utc_now(),
        )
        session = await self._get_session()
        async with session.begin():
            session.add(run)
            await session.flush()
            await session.refresh(run)
        return run

    async def complete_run(
        self,
        run_id: str,
        *,
        status: str,
        metrics_json: Optional[Dict[str, Any]] = None,
        error_json: Optional[Dict[str, Any]] = None,
        completed_at: Optional[datetime] = None,
    ) -> Optional[AgentRun]:
        session = await self._get_session()
        async with session.begin():
            run = await session.get(AgentRun, run_id)
            if not run:
                return None
            run.status = status
            run.completed_at = completed_at or utc_now()
            replace_jsonb_field(run, "metrics_json", metrics_json or {})
            if error_json is not None:
                replace_jsonb_field(run, "error_json", error_json)
            await session.flush()
            await session.refresh(run)
            return run
