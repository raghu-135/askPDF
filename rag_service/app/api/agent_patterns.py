from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.agent.tool_registry import tool_contracts_by_id
from app.agent_patterns.graph import normalize_hitl_policy_for_thread_settings
from app.agent_patterns.node_catalog import get_node_catalog
from app.agent_patterns.repository import AgentPatternRepository, AgentRunInterruptError
from app.agent_patterns.route_registry import get_route_function_registry
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


router = APIRouter(tags=["agent-workflows"])


def _new_custom_workflow_id() -> str:
    return f"custom_wf_{uuid.uuid4().hex}"


class TemplateValidationRequest(BaseModel):
    spec: Dict[str, Any] = Field(default_factory=dict)


class ThreadAgentConfigValidationRequest(BaseModel):
    overrides: Dict[str, Any] = Field(default_factory=dict)


class InternalAgentWorkflowSaveRequest(BaseModel):
    workflow_id: Optional[str] = Field(default=None, min_length=1)
    name: str = Field(..., min_length=1)
    description: str = ""
    spec_json: Dict[str, Any] = Field(default_factory=dict)


class AgentRunResumeRequest(BaseModel):
    action: str = Field(..., min_length=1)
    interrupt_id: str = Field(..., min_length=1)
    edited_payload: Optional[Dict[str, Any]] = None
    client_metadata: Optional[Dict[str, Any]] = None
    selected_option_ids: Optional[list[str]] = None
    resume_token: Optional[str] = None
    resume_version: Optional[int] = None
    thread_id: str = Field(..., min_length=1)


def _workflow_payload(workflow) -> Dict[str, Any]:
    return {
        "id": workflow.id,
        "workflow_id": workflow.id,
        "name": workflow.name,
        "description": workflow.description,
        "visibility": workflow.visibility,
        "is_builtin": workflow.is_builtin,
        "created_at": iso_utc_z(workflow.created_at) if workflow.created_at else None,
        "updated_at": iso_utc_z(workflow.updated_at) if workflow.updated_at else None,
    }


def _workflow_spec_payload(workflow) -> Dict[str, Any]:
    try:
        validation = TemplateValidator().report(workflow.spec_json if isinstance(workflow.spec_json, dict) else {})
    except Exception as exc:
        validation = {
            "valid": False,
            "errors": [f"validation failed: {exc}"],
            "warnings": [],
            "schema_version": getattr(workflow, "schema_version", None),
            "pattern_type": None,
        }
    return {
        "workflow_id": workflow.id,
        "schema_version": workflow.schema_version,
        "spec_json": workflow.spec_json if isinstance(workflow.spec_json, dict) else {},
        "validation": validation,
        "validation_result_json": workflow.validation_result_json if isinstance(workflow.validation_result_json, dict) else {},
        "created_at": iso_utc_z(workflow.created_at) if workflow.created_at else None,
        "updated_at": iso_utc_z(workflow.updated_at) if workflow.updated_at else None,
    }


def _is_compatible_workflow(workflow) -> bool:
    if not workflow or workflow.schema_version != 2 or not isinstance(workflow.spec_json, dict):
        return False
    try:
        TemplateValidator().validate(workflow.spec_json)
    except Exception:
        return False
    return True


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
        "workflow_id": run.workflow_id,
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
        "workflow_id": run.workflow_id,
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


def _capabilities_for_workflow(workflow_id: str) -> Dict[str, Any]:
    if workflow_id == EVALUATOR_REPLANNER_RAG_AGENT_ID:
        return {
            "required_tool_ids": sorted(EVALUATOR_REPLANNER_RAG_REQUIRED_TOOL_IDS),
            "node_tool_requirements": dict(sorted(EVALUATOR_REPLANNER_RAG_NODE_TOOL_REQUIREMENTS.items())),
        }
    if workflow_id == PLAN_EXECUTE_RAG_AGENT_ID:
        return {
            "required_tool_ids": sorted(PLAN_EXECUTE_RAG_REQUIRED_TOOL_IDS),
            "node_tool_requirements": dict(sorted(PLAN_EXECUTE_RAG_NODE_TOOL_REQUIREMENTS.items())),
        }
    return {
        "required_tool_ids": sorted(ROUTER_RAG_REQUIRED_TOOL_IDS),
        "node_tool_requirements": dict(sorted(ROUTER_RAG_NODE_TOOL_REQUIREMENTS.items())),
    }


def _agent_workflow_tool_contract_catalog() -> Dict[str, Any]:
    contracts: Dict[str, Any] = {}
    for contract_id, records in sorted(tool_contracts_by_id().items()):
        canonical_tools = sorted(
            str(record.get("tool_name"))
            for record in records
            if isinstance(record.get("tool_name"), str) and record.get("tool_name")
        )
        first = records[0] if records else {}
        contracts[contract_id] = {
            "id": contract_id,
            "category": first.get("category"),
            "display_name": first.get("display_name"),
            "description": first.get("description"),
            "canonical_tools": canonical_tools,
            "allowed_node_types": sorted(
                {
                    str(node_type)
                    for record in records
                    for node_type in record.get("allowed_node_types", [])
                    if node_type
                }
            ),
            "required_node_capabilities": sorted(
                {
                    str(capability)
                    for record in records
                    for capability in record.get("required_node_capabilities", [])
                    if capability
                }
            ),
            "artifact_keys": sorted(
                {
                    str(artifact_key)
                    for record in records
                    for artifact_key in record.get("artifact_keys", [])
                    if artifact_key
                }
            ),
            "warning_codes": sorted(
                {
                    str(warning_code)
                    for record in records
                    for warning_code in record.get("warning_codes", [])
                    if warning_code
                }
            ),
        }
    return contracts


@router.get("/agent-workflows")
async def list_agent_workflows():
    repo = AgentPatternRepository()
    await repo.seed_builtin_workflows()
    workflows = await repo.list_workflows(include_custom=True)
    compatible_workflows = []
    for workflow in workflows:
        try:
            if _is_compatible_workflow(workflow):
                compatible_workflows.append(workflow)
        except Exception:
            continue
    return {"agent_workflows": [_workflow_payload(workflow) for workflow in compatible_workflows]}


@router.post("/agent-workflows/validate")
async def validate_agent_workflow(req: TemplateValidationRequest):
    validator = TemplateValidator()
    return validator.report(req.spec)


@router.get("/agent-workflows/{workflow_id}")
async def get_agent_workflow(workflow_id: str):
    repo = AgentPatternRepository()
    await repo.seed_builtin_workflows()
    include_custom = workflow_id not in SUPPORTED_BUILTIN_TEMPLATE_IDS
    workflow = await repo.get_workflow(workflow_id, include_custom=include_custom)
    if not workflow or not _is_compatible_workflow(workflow):
        raise HTTPException(status_code=404, detail="Agent workflow not found")
    return {
        "agent_workflow": _workflow_payload(workflow),
        "spec": _workflow_spec_payload(workflow),
        "capabilities": _capabilities_for_workflow(workflow.id),
    }


@router.post("/internal/agent-workflows")
async def save_internal_agent_workflow(req: InternalAgentWorkflowSaveRequest):
    repo = AgentPatternRepository()
    try:
        workflow_id = (req.workflow_id or "").strip() or _new_custom_workflow_id()
        spec_json = dict(req.spec_json)
        spec_json["pattern_type"] = workflow_id
        workflow = await repo.save_custom_workflow(
            workflow_id=workflow_id,
            name=req.name,
            description=req.description,
            spec_json=spec_json,
        )
    except (TemplateValidationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "agent_workflow": _workflow_payload(workflow),
        "spec": _workflow_spec_payload(workflow),
    }


@router.delete("/internal/agent-workflows/{workflow_id}")
async def delete_internal_agent_workflow(workflow_id: str):
    repo = AgentPatternRepository()
    try:
        workflow = await repo.mark_custom_workflow_deleted(workflow_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if workflow is None:
        raise HTTPException(status_code=404, detail="Internal agent workflow not found")
    return {"status": "deleted", "agent_workflow": _workflow_payload(workflow)}


@router.get("/internal/agent-workflows/catalog")
async def get_internal_agent_workflow_catalog():
    return {
        "schema_version": 1,
        "spec_schema_version": 2,
        "graph_spec": {
            "required_schema_version": 2,
            "requires_explicit_route_fn": True,
            "reserved_node_ids": ["START", "END"],
            "start_node": "START",
            "end_node": "END",
        },
        "node_catalog": get_node_catalog(),
        "route_functions": get_route_function_registry(),
        "tool_contracts": _agent_workflow_tool_contract_catalog(),
        "defaults": {
            "context_policy": {
                "evidence_packet_limit": 12,
                "evidence_packet_content_limit": 2000,
                "final_prompt_assembly": "legacy_evidence",
            },
            "loop_policy": {
                "default_max_node_visits": 1,
            },
        },
    }


@router.get("/internal/agent-workflows/{workflow_id}")
async def get_internal_agent_workflow(workflow_id: str):
    repo = AgentPatternRepository()
    workflow = await repo.get_workflow(workflow_id, include_custom=True)
    if not workflow or workflow.is_builtin:
        raise HTTPException(status_code=404, detail="Internal agent workflow not found")
    return {
        "agent_workflow": _workflow_payload(workflow),
        "spec": _workflow_spec_payload(workflow),
    }


@router.post("/threads/{thread_id}/agent-config/validate")
async def validate_thread_agent_config(thread_id: str, req: ThreadAgentConfigValidationRequest):
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    repo = AgentPatternRepository()
    await repo.seed_builtin_workflows()
    thread_settings = await get_thread_settings(thread_id)
    agent_settings = thread_settings.get("agent_workflow") if isinstance(thread_settings, dict) else None
    agent_settings = agent_settings if isinstance(agent_settings, dict) else {}
    workflow_id = agent_settings.get("workflow_id") or ROUTER_RAG_AGENT_ID

    workflow = await repo.get_workflow(
        workflow_id,
        include_custom=workflow_id not in SUPPORTED_BUILTIN_TEMPLATE_IDS,
    )
    if not workflow:
        raise HTTPException(status_code=404, detail="Agent workflow not found")

    resolver = TemplateResolver()
    try:
        resolved_spec = resolver.resolve(
            workflow.spec_json,
            thread_settings=thread_settings,
            request_overrides=req.overrides,
        )
    except TemplateValidationError as exc:
        candidate = dict(workflow.spec_json or {})
        candidate_config = dict(candidate.get("config") or {})
        for source in (thread_settings or {}, req.overrides or {}):
            for key in ALLOWED_ROUTER_RAG_CONFIG_KEYS:
                value = source.get(key) if isinstance(source, dict) else None
                if value is not None:
                    candidate_config[key] = value
        candidate["config"] = candidate_config
        try:
            report = TemplateValidator().report(candidate)
        except Exception as report_exc:
            report = {"valid": False, "errors": [str(report_exc)], "warnings": []}
        report["errors"] = report["errors"] or [str(exc)]
        return {
            "valid": False,
            "workflow_id": workflow.id,
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
        "workflow_id": workflow.id,
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
