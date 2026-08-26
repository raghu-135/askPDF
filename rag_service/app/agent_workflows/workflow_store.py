from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional

from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.agent_workflows.builtin_workflows import builtin_workflow_keys, load_builtin_workflows
from app.runtime.langgraph.validator import WorkflowValidationError
from app.runtime.builder_registry import BuilderSelectionError, builder_for_definition
from app.runtime.contracts import AgentDefinition
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
    async with session.begin():
        for workflow_def in load_builtin_workflows():
            spec_json = workflow_def["spec_json"]
            builtin_key = workflow_def["builtin_key"]
            framework = str(workflow_def.get("framework") or "").strip()
            builder_id = str(workflow_def.get("builder_id") or "").strip()
            if not framework or not builder_id:
                raise ValueError(f"Builtin workflow {builtin_key} is missing runtime identity")
            definition = AgentDefinition(
                definition_id=builtin_key,
                framework=framework,
                builder_id=builder_id,
                category=workflow_def.get("category"),
                display_name=workflow_def.get("name"),
            )
            try:
                provider = builder_for_definition(definition)
                validation = await provider.validate(definition, spec_json)
            except BuilderSelectionError as exc:
                raise WorkflowValidationError(str(exc)) from exc
            validation_result = {
                "valid": validation.valid,
                "errors": [issue.message for issue in validation.issues],
            }
            if not validation.valid:
                raise WorkflowValidationError(
                    "; ".join(validation_result["errors"]) or f"Invalid workflow: {builtin_key}"
                )
            metadata = {
                "source": WorkflowVisibility.BUILTIN.value,
                "builtin_key": builtin_key,
                "framework": framework,
                "builder_id": builder_id,
                "category": workflow_def.get("category"),
                "version": 1,
                "version_id": f"{builtin_key}:v1",
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
                    framework=framework,
                    builder_id=builder_id,
                    category=workflow_def.get("category"),
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
                workflow.framework = framework
                workflow.builder_id = builder_id
                workflow.category = workflow_def.get("category")
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
    framework: Optional[str] = None,
    builder_id: Optional[str] = None,
    description: str = "",
    visibility: str = WorkflowVisibility.INTERNAL.value,
) -> AgentWorkflow:
    """Create or update a mutable internal/custom workflow spec."""
    if workflow_id in builtin_workflow_keys():
        raise ValueError("built-in agent workflows cannot be authored through the internal path")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if not isinstance(spec_json, dict):
        raise WorkflowValidationError("spec must be an object")
    if spec_json.get("schema_version") != 1:
        raise WorkflowValidationError("internal custom agent workflow specs must use schema_version 1")

    async with session.begin():
        workflow = await session.get(AgentWorkflow, workflow_id) if workflow_id else None
        existing_named_workflow = await get_workflow_by_name(session, name)
        if existing_named_workflow is not None and (workflow is None or existing_named_workflow.id != workflow.id):
            raise ValueError(f"agent workflow name already exists: {name}")
        previous_metadata = workflow.metadata_json if workflow and isinstance(workflow.metadata_json, dict) else {}
        stored_framework = str(getattr(workflow, "framework", "") or "") if workflow is not None else ""
        stored_builder_id = str(getattr(workflow, "builder_id", "") or "") if workflow is not None else ""
        if workflow is not None and framework is not None and framework != stored_framework:
            raise ValueError("workflow framework identity is immutable")
        if workflow is not None and builder_id is not None and builder_id != stored_builder_id:
            raise ValueError("workflow builder identity is immutable")
        next_version = 1
        workflow_key = workflow_id or spec_json.get("workflow_id") or name
        framework = stored_framework or str(framework or "").strip()
        builder_id = stored_builder_id or str(builder_id or "").strip()
        if not framework or not builder_id:
            raise ValueError("Agent workflow runtime identity is required")
        definition = AgentDefinition(
            definition_id=str(workflow_key),
            framework=framework,
            builder_id=builder_id,
            display_name=name,
        )
        try:
            provider = builder_for_definition(definition)
            capabilities = await provider.capabilities(definition)
            if workflow is None and not capabilities.authoring:
                raise ValueError(f"runtime_capability_unsupported: {framework}/{builder_id} does not support authoring")
            validation = await provider.validate(definition, spec_json)
            if not validation.valid:
                raise WorkflowValidationError(
                    "; ".join(issue.message for issue in validation.issues) or "Invalid workflow"
                )
            normalized_spec = dict(await provider.normalize(definition, spec_json))
        except BuilderSelectionError as exc:
            raise WorkflowValidationError(str(exc)) from exc
        validation_result = {
            "valid": validation.valid,
            "errors": [issue.message for issue in validation.issues],
        }
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
                schema_version=1,
                framework=framework,
                builder_id=builder_id,
                spec_json=normalized_spec,
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
            workflow.schema_version = 1
            workflow.framework = framework
            workflow.builder_id = builder_id
            replace_jsonb_field(workflow, "spec_json", normalized_spec)
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
    framework: Optional[str] = None,
    builder_id: Optional[str] = None,
    description: str = "",
    visibility: str = WorkflowVisibility.INTERNAL.value,
    changelog: str = "",
) -> tuple[AgentWorkflow, AgentWorkflowVersion]:
    workflow = await save_custom_workflow(
        session,
        workflow_id=workflow_id,
        name=name,
        description=description,
        visibility=visibility,
        spec_json=spec_json,
        framework=framework,
        builder_id=builder_id,
    )
    return workflow, workflow_version(workflow)
