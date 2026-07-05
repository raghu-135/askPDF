from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.agent_patterns.debug_trace import build_debug_trace
from app.agent_patterns.repository import AgentPatternRepository
from app.agent_patterns.templates import (
    ALLOWED_ROUTER_RAG_CONFIG_KEYS,
    PLAN_EXECUTE_RAG_AGENT_ID,
    PLAN_EXECUTE_RAG_NODE_TOOL_REQUIREMENTS,
    PLAN_EXECUTE_RAG_REQUIRED_TOOL_IDS,
    ROUTER_RAG_AGENT_ID,
    ROUTER_RAG_NODE_TOOL_REQUIREMENTS,
    ROUTER_RAG_REQUIRED_TOOL_IDS,
    SUPPORTED_BUILTIN_TEMPLATE_IDS,
)
from app.agent_patterns.validator import TemplateResolver, TemplateValidationError, TemplateValidator
from app.db import get_thread, get_thread_settings
from app.time_utils import iso_utc_z


router = APIRouter(tags=["agent-patterns"])


class TemplateValidationRequest(BaseModel):
    spec: Dict[str, Any] = Field(default_factory=dict)


class ThreadAgentConfigValidationRequest(BaseModel):
    overrides: Dict[str, Any] = Field(default_factory=dict)


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
    validator = TemplateValidator()
    return {
        "id": version.id,
        "template_id": version.template_id,
        "version": version.version,
        "schema_version": version.schema_version,
        "spec_json": version.spec_json,
        "validation": validator.report(version.spec_json),
        "validation_result_json": version.validation_result_json,
        "changelog": version.changelog,
        "created_at": iso_utc_z(version.created_at) if version.created_at else None,
    }


def _trace_for_response(
    trace: Dict[str, Any],
    *,
    chat_turn_id: str | None,
    metrics: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    response_trace = deepcopy(trace)
    trace_metrics = response_trace.get("metrics") if isinstance(response_trace.get("metrics"), dict) else {}
    response_trace["metrics"] = {**trace_metrics, **(metrics or {})}
    response_trace["chat_turn_id"] = chat_turn_id

    attributes = dict(response_trace.get("attributes") or {})
    attributes["askpdf.chat_turn.id"] = chat_turn_id
    response_trace["attributes"] = attributes

    for span in response_trace.get("spans") or []:
        if isinstance(span, dict) and span.get("parent_span_id") is None:
            span_attributes = dict(span.get("attributes") or {})
            span_attributes["askpdf.chat_turn.id"] = chat_turn_id
            span["attributes"] = span_attributes
            break
    return response_trace


def _run_payload(run, *, chat_turn=None) -> Dict[str, Any]:
    payload = {
        "id": run.id,
        "thread_id": run.thread_id,
        "user_id": run.user_id,
        "template_id": run.template_id,
        "template_version_id": run.template_version_id,
        "chat_turn_id": run.chat_turn_id,
        "resolved_spec_json": run.resolved_spec_json,
        "status": run.status,
        "checkpoint_thread_id": run.checkpoint_thread_id,
        "started_at": iso_utc_z(run.started_at) if run.started_at else None,
        "completed_at": iso_utc_z(run.completed_at) if run.completed_at else None,
        "error_json": run.error_json,
        "metrics_json": run.metrics_json,
    }
    if chat_turn is not None:
        turn_payload = chat_turn.payload if isinstance(chat_turn.payload, dict) else {}
        metadata = turn_payload.get("metadata") if isinstance(turn_payload.get("metadata"), dict) else {}
        metrics = run.metrics_json if isinstance(run.metrics_json, dict) else {}
        trace = metadata.get("agent_debug_trace") if isinstance(metadata.get("agent_debug_trace"), dict) else None
        if trace is not None:
            payload["debug"] = {
                "trace": _trace_for_response(trace, chat_turn_id=chat_turn.id, metrics=metrics),
            }
    elif run.status == "failed":
        metrics = run.metrics_json if isinstance(run.metrics_json, dict) else {}
        trace = build_debug_trace(
            run=run,
            chat_turn=None,
            node_events=[],
            tool_events=[],
            metrics=metrics,
            route=metrics.get("route"),
            route_reason=None,
            error=run.error_json,
        )
        payload["debug"] = {
            "trace": _trace_for_response(trace, chat_turn_id=None, metrics=metrics),
        }
    return payload


def _run_summary_payload(run) -> Dict[str, Any]:
    metrics = run.metrics_json if isinstance(run.metrics_json, dict) else {}
    error = run.error_json if isinstance(run.error_json, dict) else None
    return {
        "id": run.id,
        "thread_id": run.thread_id,
        "template_id": run.template_id,
        "template_version_id": run.template_version_id,
        "chat_turn_id": run.chat_turn_id,
        "status": run.status,
        "started_at": iso_utc_z(run.started_at) if run.started_at else None,
        "completed_at": iso_utc_z(run.completed_at) if run.completed_at else None,
        "metrics": {
            "duration_ms": metrics.get("duration_ms"),
            "route": metrics.get("route"),
            "node_event_count": metrics.get("node_event_count", 0),
            "tool_event_count": metrics.get("tool_event_count", 0),
            "tool_warning_count": metrics.get("tool_warning_count", 0),
            "tool_error_count": metrics.get("tool_error_count", 0),
            "error_count": metrics.get("error_count", 0),
        },
        "error": {
            "code": error.get("code"),
            "raw_message": error.get("raw_message"),
            "retryable": error.get("retryable"),
        } if error else None,
    }


def _capabilities_for_pattern(template_id: str) -> Dict[str, Any]:
    if template_id == PLAN_EXECUTE_RAG_AGENT_ID:
        return {
            "required_tool_ids": sorted(PLAN_EXECUTE_RAG_REQUIRED_TOOL_IDS),
            "node_tool_requirements": dict(sorted(PLAN_EXECUTE_RAG_NODE_TOOL_REQUIREMENTS.items())),
        }
    return {
        "required_tool_ids": sorted(ROUTER_RAG_REQUIRED_TOOL_IDS),
        "node_tool_requirements": dict(sorted(ROUTER_RAG_NODE_TOOL_REQUIREMENTS.items())),
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
        "capabilities": _capabilities_for_pattern(template.id),
    }


@router.post("/agent-patterns/validate")
async def validate_agent_pattern(req: TemplateValidationRequest):
    validator = TemplateValidator()
    return validator.report(req.spec)


@router.post("/threads/{thread_id}/agent-config/validate")
async def validate_thread_agent_config(thread_id: str, req: ThreadAgentConfigValidationRequest):
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    repo = AgentPatternRepository()
    await repo.seed_builtin_templates()
    thread_settings = await get_thread_settings(thread_id)
    agent_settings = thread_settings.get("agent_pattern") if isinstance(thread_settings, dict) else None
    agent_settings = agent_settings if isinstance(agent_settings, dict) else {}
    template_id = agent_settings.get("template_id") or ROUTER_RAG_AGENT_ID
    if template_id not in SUPPORTED_BUILTIN_TEMPLATE_IDS:
        template_id = ROUTER_RAG_AGENT_ID

    template, version = await repo.get_template_with_current_version(template_id)
    if not template or not version:
        raise HTTPException(status_code=404, detail="Agent pattern not found")

    resolver = TemplateResolver()
    try:
        resolved_spec = resolver.resolve(
            version.spec_json,
            thread_settings=thread_settings,
            request_overrides=req.overrides,
        )
    except TemplateValidationError as exc:
        candidate = dict(version.spec_json or {})
        candidate_config = dict(candidate.get("config") or {})
        for source in (thread_settings or {}, req.overrides or {}):
            for key in ALLOWED_ROUTER_RAG_CONFIG_KEYS:
                value = source.get(key) if isinstance(source, dict) else None
                if value is not None:
                    candidate_config[key] = value
        candidate["config"] = candidate_config
        report = TemplateValidator().report(candidate)
        report["errors"] = report["errors"] or [str(exc)]
        return {
            "valid": False,
            "template_id": template.id,
            "template_version": version.version,
            "template_version_id": version.id,
            "validation": report,
            "resolved_spec_json": candidate,
        }

    return {
        "valid": True,
        "template_id": template.id,
        "template_version": version.version,
        "template_version_id": version.id,
        "validation": TemplateValidator().report(resolved_spec),
        "resolved_spec_json": resolved_spec,
    }


@router.get("/threads/{thread_id}/agent-runs")
async def list_thread_agent_runs(thread_id: str, limit: int = Query(20, ge=1, le=100)):
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    repo = AgentPatternRepository()
    runs = await repo.list_runs_for_thread(thread_id, limit=limit)
    return {
        "thread_id": thread_id,
        "limit": limit,
        "agent_runs": [_run_summary_payload(run) for run in runs],
    }


@router.get("/agent-runs/{run_id}")
async def get_agent_run(run_id: str):
    repo = AgentPatternRepository()
    run = await repo.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Agent run not found")
    chat_turn = await repo.get_chat_turn_for_run(run)
    return {"agent_run": _run_payload(run, chat_turn=chat_turn)}
