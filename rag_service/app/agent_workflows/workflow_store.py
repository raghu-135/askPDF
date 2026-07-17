from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.agent_workflows.builtin_workflows import builtin_workflow_keys, load_builtin_workflows
from app.agent_workflows.validator import WorkflowValidationError, WorkflowValidator
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import AgentWorkflow, WorkflowVisibility
from app.time_utils import utc_now


@dataclass
class AgentWorkflowVersion:
    id: str
    workflow_id: str
    version: int
    schema_version: int
    spec_json: Dict[str, Any]
    validation_result_json: Dict[str, Any]
    metadata_json: Dict[str, Any]


def workflow_version(workflow: AgentWorkflow) -> AgentWorkflowVersion:
    metadata = workflow.metadata_json if isinstance(workflow.metadata_json, dict) else {}
    version = workflow.version
    version_id = str(metadata.get("version_id") or f"{workflow.id}:v{version}")
    return AgentWorkflowVersion(
        id=version_id,
        workflow_id=workflow.id,
        version=version,
        schema_version=workflow.schema_version,
        spec_json=workflow.spec_json,
        validation_result_json=workflow.validation_result_json,
        metadata_json=metadata,
    )


async def get_workflow_by_name(session: AsyncSession, name: str) -> Optional[AgentWorkflow]:
    result = await session.execute(select(AgentWorkflow).where(AgentWorkflow.name == name))
    return result.scalars().first()


async def get_workflow_by_builtin_key(session: AsyncSession, builtin_key: str) -> Optional[AgentWorkflow]:
    result = await session.execute(
        select(AgentWorkflow).where(AgentWorkflow.metadata_json["builtin_key"].astext == builtin_key)
    )
    workflow = result.scalars().first()
    if workflow is not None:
        return workflow
    result = await session.execute(
        select(AgentWorkflow).where(AgentWorkflow.spec_json["workflow_id"].astext == builtin_key)
    )
    return result.scalars().first()


async def seed_builtin_workflows(session: AsyncSession) -> None:
    validator = WorkflowValidator()
    async with session.begin():
        for workflow_def in load_builtin_workflows():
            spec_json = workflow_def["spec_json"]
            validation_result = validator.validate(spec_json)
            builtin_key = workflow_def["builtin_key"]
            metadata = {
                "source": WorkflowVisibility.BUILTIN.value,
                "builtin_key": builtin_key,
                "version": spec_json.get("version") or 2,
                "version_id": f"{builtin_key}:v{spec_json.get('version') or 2}",
            }

            workflow = await get_workflow_by_builtin_key(session, builtin_key)
            if workflow is None:
                workflow = await get_workflow_by_name(session, workflow_def["name"])
            if workflow is None:
                workflow = AgentWorkflow(
                    id=builtin_key,
                    name=workflow_def["name"],
                    description=workflow_def["description"],
                    visibility=workflow_def["visibility"],
                    is_builtin=workflow_def["is_builtin"],
                    schema_version=spec_json["schema_version"],
                    spec_json=spec_json,
                    validation_result_json=validation_result,
                    metadata_json=metadata,
                )
                session.add(workflow)
            else:
                if not workflow.is_builtin and workflow.visibility != WorkflowVisibility.DELETED.value:
                    raise ValueError(f"agent workflow name already exists: {workflow_def['name']}")
                workflow.name = workflow_def["name"]
                workflow.description = workflow_def["description"]
                workflow.visibility = workflow_def["visibility"]
                workflow.is_builtin = workflow_def["is_builtin"]
                workflow.schema_version = spec_json["schema_version"]
                replace_jsonb_field(workflow, "spec_json", spec_json)
                replace_jsonb_field(workflow, "validation_result_json", validation_result)
                replace_jsonb_field(workflow, "metadata_json", metadata)
                workflow.updated_at = utc_now()


async def list_workflows(session: AsyncSession, *, include_custom: bool = False) -> list[AgentWorkflow]:
    async with session.begin():
        visibility_filter = (
            AgentWorkflow.is_builtin.is_(True)
            if not include_custom
            else or_(
                AgentWorkflow.is_builtin.is_(True),
                AgentWorkflow.visibility.in_([WorkflowVisibility.PUBLIC.value, WorkflowVisibility.INTERNAL.value]),
            )
        )
        result = await session.execute(
            select(AgentWorkflow)
            .where(visibility_filter)
            .order_by(AgentWorkflow.name.asc())
        )
        return list(result.scalars().all())


async def mark_custom_workflow_deleted(session: AsyncSession, workflow_id: str) -> Optional[AgentWorkflow]:
    async with session.begin():
        workflow = await session.get(AgentWorkflow, workflow_id)
        if workflow is None or workflow.is_builtin:
            if workflow is not None and workflow.is_builtin:
                raise ValueError("built-in agent workflows cannot be deleted")
            return None
        workflow.visibility = WorkflowVisibility.DELETED.value
        workflow.updated_at = utc_now()
        await session.flush()
        return workflow


async def get_workflow(
    session: AsyncSession,
    workflow_id: str,
    *,
    include_custom: bool = False,
) -> Optional[AgentWorkflow]:
    async with session.begin():
        workflow = await session.get(AgentWorkflow, workflow_id)
        if workflow is None and workflow_id in builtin_workflow_keys():
            workflow = await get_workflow_by_builtin_key(session, workflow_id)
        if not workflow:
            return None
        if not include_custom and not workflow.is_builtin:
            return None
        if include_custom and not workflow.is_builtin and workflow.visibility not in {WorkflowVisibility.PUBLIC.value, WorkflowVisibility.INTERNAL.value}:
            return None
        return workflow


async def get_workflow_with_current_version(
    session: AsyncSession,
    workflow_id: str,
    *,
    include_custom: bool = False,
) -> tuple[Optional[AgentWorkflow], Optional[AgentWorkflowVersion]]:
    workflow = await get_workflow(session, workflow_id, include_custom=include_custom)
    if workflow is None:
        return None, None
    return workflow, workflow_version(workflow)


async def get_workflow_version(
    session: AsyncSession,
    workflow_id: str,
    version: int,
    *,
    include_custom: bool = False,
) -> tuple[Optional[AgentWorkflow], Optional[AgentWorkflowVersion]]:
    workflow, current_version = await get_workflow_with_current_version(
        session,
        workflow_id,
        include_custom=include_custom,
    )
    if current_version is None or current_version.version != int(version):
        return None, None
    return workflow, current_version


async def save_custom_workflow(
    session: AsyncSession,
    *,
    workflow_id: Optional[str],
    name: str,
    spec_json: Dict[str, Any],
    description: str = "",
    visibility: str = WorkflowVisibility.INTERNAL.value,
    increment_version: bool = True,
) -> AgentWorkflow:
    """Create or update a mutable internal/custom workflow spec."""
    if workflow_id in builtin_workflow_keys():
        raise ValueError("built-in agent workflows cannot be authored through the internal path")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if not isinstance(spec_json, dict):
        raise WorkflowValidationError("spec must be an object")
    if spec_json.get("schema_version") != 2:
        raise WorkflowValidationError("internal custom agent workflow specs must use schema_version 2")

    validation_result = WorkflowValidator().validate(spec_json)
    async with session.begin():
        workflow = await session.get(AgentWorkflow, workflow_id) if workflow_id else None
        existing_named_workflow = await get_workflow_by_name(session, name)
        if existing_named_workflow is not None and (workflow is None or existing_named_workflow.id != workflow.id):
            raise ValueError(f"agent workflow name already exists: {name}")
        previous_metadata = workflow.metadata_json if workflow and isinstance(workflow.metadata_json, dict) else {}
        previous_version = previous_metadata.get("version")
        try:
            next_version = int(previous_version) + 1 if workflow is not None and increment_version else int(previous_version or 1)
        except (TypeError, ValueError):
            next_version = 1
        workflow_key = workflow_id or spec_json.get("workflow_id") or name
        metadata = {
            **previous_metadata,
            "source": "custom",
            "version": next_version,
            "version_id": f"{workflow_key}:v{next_version}",
        }
        if workflow is None:
            workflow = AgentWorkflow(
                id=workflow_id or str(uuid.uuid4()),
                name=name,
                description=description,
                visibility=visibility,
                is_builtin=False,
                schema_version=2,
                spec_json=spec_json,
                validation_result_json=validation_result,
                metadata_json=metadata,
            )
            session.add(workflow)
        else:
            if workflow.is_builtin:
                raise ValueError("built-in agent workflows cannot be authored through the internal path")
            workflow.name = name
            workflow.description = description
            workflow.visibility = visibility
            workflow.is_builtin = False
            workflow.schema_version = 2
            replace_jsonb_field(workflow, "spec_json", spec_json)
            replace_jsonb_field(workflow, "validation_result_json", validation_result)
            replace_jsonb_field(workflow, "metadata_json", metadata)
            workflow.updated_at = utc_now()

        await session.flush()
        return workflow


async def save_internal_workflow_version(
    session: AsyncSession,
    *,
    workflow_id: str,
    name: str,
    spec_json: Dict[str, Any],
    description: str = "",
    visibility: str = WorkflowVisibility.INTERNAL.value,
    changelog: str = "",
    increment_version: bool = True,
) -> tuple[AgentWorkflow, AgentWorkflowVersion]:
    workflow = await save_custom_workflow(
        session,
        workflow_id=workflow_id,
        name=name,
        description=description,
        visibility=visibility,
        spec_json=spec_json,
        increment_version=increment_version,
    )
    return workflow, workflow_version(workflow)
