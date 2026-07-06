from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.agent_patterns.debug_trace import build_debug_graph
from app.agent_patterns.graph import normalize_hitl_policy_for_thread_settings
from app.agent_patterns.repository import AgentPatternRepository, AgentRunInterruptError
from app.agent_patterns.service import AgentRunService
from app.agent_patterns.templates import (
    ALLOWED_ROUTER_RAG_CONFIG_KEYS,
    EVALUATOR_REPLANNER_RAG_AGENT_ID,
    EVALUATOR_REPLANNER_RAG_NODE_TOOL_REQUIREMENTS,
    EVALUATOR_REPLANNER_RAG_REQUIRED_TOOL_IDS,
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


class InternalAgentPatternCreateRequest(BaseModel):
    template_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = ""
    owner_id: Optional[str] = None
    version: Optional[int] = Field(default=None, ge=1)
    changelog: Optional[str] = None
    spec_json: Dict[str, Any] = Field(default_factory=dict)
    set_current: bool = True


class AgentRunResumeRequest(BaseModel):
    action: str = Field(..., min_length=1)
    interrupt_id: str = Field(..., min_length=1)
    edited_payload: Optional[Dict[str, Any]] = None
    client_metadata: Optional[Dict[str, Any]] = None
    selected_option_ids: Optional[list[str]] = None
    resume_token: Optional[str] = None
    resume_version: Optional[int] = None
    thread_id: str = Field(..., min_length=1)


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


def _debug_payload_for_response(run) -> Dict[str, Any] | None:
    debug = run.debug_trace_json if isinstance(run.debug_trace_json, dict) else None
    if not debug or debug.get("version") != 1:
        return None
    trace = debug.get("trace") if isinstance(debug.get("trace"), dict) else None
    summary = debug.get("summary") if isinstance(debug.get("summary"), dict) else None
    if trace is None or summary is None:
        return None
    return {
        **debug,
        "trace": trace,
        "summary": summary,
        "graph": build_debug_graph(
            resolved_spec=run.resolved_spec_json if isinstance(run.resolved_spec_json, dict) else {},
            summary=summary,
        ),
    }


def _turn_summary_payload(turn) -> Dict[str, Any]:
    trace_refs = turn.agent_trace_refs_json if isinstance(turn.agent_trace_refs_json, dict) else {}
    return {
        "id": turn.id,
        "kind": turn.agent_run_turn_kind,
        "sequence": turn.agent_run_sequence,
        "trace_refs": trace_refs,
    }


def _pending_interrupt_payload(run) -> Dict[str, Any] | None:
    pending = run.pending_interrupt_json if isinstance(run.pending_interrupt_json, dict) else None
    return dict(pending) if pending else None


def _run_payload(run, turns=None) -> Dict[str, Any]:
    turns = turns or []
    payload = {
        "id": run.id,
        "thread_id": run.thread_id,
        "user_id": run.user_id,
        "template_id": run.template_id,
        "template_version_id": run.template_version_id,
        "turns": [_turn_summary_payload(turn) for turn in turns],
        "resolved_spec_json": run.resolved_spec_json,
        "status": run.status,
        "checkpoint_thread_id": run.checkpoint_thread_id,
        "pending_interrupt": _pending_interrupt_payload(run),
        "started_at": iso_utc_z(run.started_at) if run.started_at else None,
        "completed_at": iso_utc_z(run.completed_at) if run.completed_at else None,
        "error_json": run.error_json,
        "metrics_json": run.metrics_json,
        "debug": _debug_payload_for_response(run),
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
        "status": run.status,
        "pending_interrupt": _pending_interrupt_payload(run),
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
            "replan_count": metrics.get("replan_count", 0),
            "evaluation_confidence": metrics.get("evaluation_confidence"),
        },
        "error": {
            "code": error.get("code"),
            "raw_message": error.get("raw_message"),
            "retryable": error.get("retryable"),
        } if error else None,
    }


def _capabilities_for_pattern(template_id: str) -> Dict[str, Any]:
    if template_id == EVALUATOR_REPLANNER_RAG_AGENT_ID:
        return {
            "required_tool_ids": sorted(EVALUATOR_REPLANNER_RAG_REQUIRED_TOOL_IDS),
            "node_tool_requirements": dict(sorted(EVALUATOR_REPLANNER_RAG_NODE_TOOL_REQUIREMENTS.items())),
        }
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


@router.post("/internal/agent-patterns")
async def create_internal_agent_pattern(req: InternalAgentPatternCreateRequest):
    repo = AgentPatternRepository()
    try:
        template, version = await repo.create_internal_template_version(
            template_id=req.template_id,
            name=req.name,
            description=req.description,
            owner_id=req.owner_id,
            version=req.version,
            changelog=req.changelog,
            spec_json=req.spec_json,
            set_current=req.set_current,
        )
    except (TemplateValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "agent_pattern": _template_payload(template),
        "version": _version_payload(version),
    }


@router.get("/internal/agent-patterns/{template_id}")
async def get_internal_agent_pattern(template_id: str):
    repo = AgentPatternRepository()
    template, version = await repo.get_template_with_current_version(template_id, include_custom=True)
    if not template or not version or template.is_builtin:
        raise HTTPException(status_code=404, detail="Internal agent pattern not found")
    return {
        "agent_pattern": _template_payload(template),
        "current_version": _version_payload(version),
    }


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

    resolved_config = resolved_spec.get("config") if isinstance(resolved_spec.get("config"), dict) else {}
    resolved_config["hitl_policy"] = normalize_hitl_policy_for_thread_settings(
        resolved_config.get("hitl_policy"),
        thread_settings,
    )
    resolved_spec["config"] = resolved_config
    return {
        "valid": True,
        "template_id": template.id,
        "template_version": version.version,
        "template_version_id": version.id,
        "validation": TemplateValidator().report(resolved_spec),
        "resolved_spec_json": resolved_spec,
    }


@router.get("/threads/{thread_id}/agent-runs")
async def list_thread_agent_runs(
    thread_id: str,
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
):
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    repo = AgentPatternRepository()
    runs = await repo.list_runs_for_thread(thread_id, limit=limit, status=status)
    return {
        "thread_id": thread_id,
        "limit": limit,
        "status": status,
        "agent_runs": [_run_summary_payload(run) for run in runs],
    }


@router.get("/agent-runs/{run_id}")
async def get_agent_run(
    run_id: str,
    thread_id: str = Query(..., min_length=1),
):
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Agent run not found")

    repo = AgentPatternRepository()
    run = await repo.get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail="Agent run not found")
    turns = await repo.list_chat_turns_for_run(run.id)
    return {"agent_run": _run_payload(run, turns)}


@router.post("/agent-runs/{run_id}/resume")
async def resume_agent_run(run_id: str, req: AgentRunResumeRequest):
    thread = await get_thread(req.thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Agent run not found")

    service = AgentRunService()
    try:
        result = await service.resume_agent_run(
            run_id,
            interrupt_id=req.interrupt_id,
            action=req.action,
            edited_payload=req.edited_payload,
            client_metadata=req.client_metadata,
            selected_option_ids=req.selected_option_ids,
            resume_token=req.resume_token,
            resume_version=req.resume_version,
            expected_thread_id=req.thread_id,
        )
    except AgentRunInterruptError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    repo = AgentPatternRepository()
    turns = await repo.list_chat_turns_for_run(result.run.id)
    return {
        "agent_run": _run_payload(result.run, turns),
        "interrupt": result.interrupt,
        "outcome": result.outcome,
        "duplicate": result.duplicate,
    }
