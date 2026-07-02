from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agent_patterns.repository import AgentPatternRepository
from app.agent_patterns.validator import TemplateValidator
from app.time_utils import iso_utc_z


router = APIRouter(tags=["agent-patterns"])


class TemplateValidationRequest(BaseModel):
    spec: Dict[str, Any] = Field(default_factory=dict)


def _template_payload(template) -> Dict[str, Any]:
    return {
        "id": template.id,
        "name": template.name,
        "description": template.description,
        "visibility": template.visibility,
        "owner_id": template.owner_id,
        "current_version_id": template.current_version_id,
        "is_builtin": template.is_builtin,
        "created_at": iso_utc_z(template.created_at) if template.created_at else None,
        "updated_at": iso_utc_z(template.updated_at) if template.updated_at else None,
    }


def _version_payload(version) -> Dict[str, Any]:
    return {
        "id": version.id,
        "template_id": version.template_id,
        "version": version.version,
        "schema_version": version.schema_version,
        "spec_json": version.spec_json,
        "validation_result_json": version.validation_result_json,
        "changelog": version.changelog,
        "created_at": iso_utc_z(version.created_at) if version.created_at else None,
    }


def _run_payload(run) -> Dict[str, Any]:
    return {
        "id": run.id,
        "thread_id": run.thread_id,
        "user_id": run.user_id,
        "template_id": run.template_id,
        "template_version_id": run.template_version_id,
        "resolved_spec_json": run.resolved_spec_json,
        "status": run.status,
        "checkpoint_thread_id": run.checkpoint_thread_id,
        "started_at": iso_utc_z(run.started_at) if run.started_at else None,
        "completed_at": iso_utc_z(run.completed_at) if run.completed_at else None,
        "error_json": run.error_json,
        "metrics_json": run.metrics_json,
    }


@router.get("/agent-patterns")
async def list_agent_patterns():
    repo = AgentPatternRepository()
    await repo.seed_builtin_templates()
    templates = await repo.list_templates()
    return {"agent_patterns": [_template_payload(template) for template in templates]}


@router.get("/agent-patterns/{template_id}")
async def get_agent_pattern(template_id: str):
    repo = AgentPatternRepository()
    await repo.seed_builtin_templates()
    template, version = await repo.get_template_with_current_version(template_id)
    if not template or not version:
        raise HTTPException(status_code=404, detail="Agent pattern not found")
    return {
        "agent_pattern": _template_payload(template),
        "current_version": _version_payload(version),
    }


@router.post("/agent-patterns/validate")
async def validate_agent_pattern(req: TemplateValidationRequest):
    validator = TemplateValidator()
    errors = validator.collect_errors(req.spec)
    return {
        "valid": not errors,
        "errors": errors,
    }


@router.get("/agent-runs/{run_id}")
async def get_agent_run(run_id: str):
    run = await AgentPatternRepository().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return {"agent_run": _run_payload(run)}
