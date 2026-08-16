from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.tool_registry import tool_contracts_by_id
from app.agent_workflows.node_catalog import get_node_catalog
from app.agent_workflows.repository import AgentWorkflowRepository, AgentRunInterruptError
from app.agent_workflows.route_registry import get_route_function_registry
from app.agent_workflows.service import AgentRunService
from app.agent_workflows.execution_stream import AgentExecutionEventSink, retain_background_task
from app.agent_workflows.builtin_workflows import builtin_workflow_keys, load_builtin_workflows
from app.agent_workflows.parallel_contracts import parallel_policy_catalog
from app.agent_workflows.corrective_contracts import corrective_policy_catalog
from app.agent_workflows.workflow_requirements import (
    workflow_node_tool_requirements,
    workflow_required_tool_ids,
)
from app.agent_workflows.workflow_runtime import (
    ALLOWED_WORKFLOW_CONFIG_KEYS,
    default_agent_workflow_key,
    with_default_runtime,
    workflow_supports_replans,
)
from app.runtime.langgraph.checkpointing import delete_agent_checkpoints, open_agent_checkpointer
from app.agent_workflows.chat_cancellation import (
    CHAT_CANCEL_AWAITING_HUMAN,
    CHAT_CANCEL_UNSUPPORTED,
)
from app.agent_workflows.trace_details import detail_manifest
from app.runtime.langgraph.studio_runtime import (
    RUN_KIND as BUILDER_TEST_RUN_KIND,
    delete_previous_builder_tests,
    latest_builder_test,
    request_builder_test_cancel,
    spec_fingerprint,
    stream_builder_test,
)
from app.runtime.catalog import catalog_payload
from app.runtime.builder_registry import BuilderSelectionError, builder_for_definition
from app.runtime.contracts import AgentDefinition
from app.db import AgentRunStatus, get_thread, get_thread_settings
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET
from app.models.requests import ThreadChatRequest
from app.services.embedding_model_service import (
    EmbeddingModelResolutionError,
    EmbeddingModelUnavailableError,
    require_thread_embedding_ready,
)
from app.time_utils import iso_utc_z


router = APIRouter(tags=["agent-workflows"])


async def request_chat_run_cancel(run_id: str, *, thread_id: str):
    """Compatibility seam for API callers; cancellation routes through the adapter registry."""

    return await AgentRunService().cancel_agent_run(run_id, thread_id=thread_id)


async def _require_ready_thread(thread_id: str):
    try:
        return await require_thread_embedding_ready(thread_id)
    except EmbeddingModelResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "embedding_model_unavailable", "message": str(exc)},
        ) from exc


class WorkflowValidationRequest(BaseModel):
    spec: Dict[str, Any] = Field(default_factory=dict)
    framework: str = "langgraph"
    builder_id: str = "langgraph_graph"


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


class AgentRunCancelRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)


class BuilderTransientMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=20000)


class BuilderTestRunRequest(ThreadChatRequest):
    builder_session_id: str = Field(..., min_length=1, max_length=200)
    base_workflow_id: str = Field(..., min_length=1)
    spec: Dict[str, Any] = Field(default_factory=dict)
    allow_external_tools: bool = False
    transient_messages: list[BuilderTransientMessage] = Field(default_factory=list, max_length=100)
    workflow_spec_fingerprint: Optional[str] = Field(default=None, max_length=128)


class BuilderTestRunResumeRequest(AgentRunResumeRequest):
    llm_model: str = Field(..., min_length=1)
    use_web_search: bool = False
    use_reranker: Optional[bool] = True
    context_window: int = DEFAULT_TOKEN_BUDGET
    replans: Optional[int] = None
    system_role_override: Optional[str] = None
    tool_instructions_override: Optional[Dict[str, str]] = None
    custom_instructions_override: Optional[str] = None
    hitl_web_approval: bool = False
    client_timezone: Optional[str] = None
    client_locale: Optional[str] = None
    client_now_iso: Optional[str] = None


def _workflow_payload(workflow) -> Dict[str, Any]:
    spec = workflow.spec_json if isinstance(workflow.spec_json, dict) else {}
    metadata = workflow.metadata_json if isinstance(workflow.metadata_json, dict) else {}
    known_builtin_keys = set(builtin_workflow_keys())
    builtin_key = None
    if workflow.is_builtin:
        metadata_key = str(metadata.get("builtin_key") or "").strip()
        spec_key = str(spec.get("workflow_id") or "").strip()
        row_key = str(workflow.id or "").strip()
        builtin_key = next(
            (key for key in (metadata_key, spec_key, row_key) if key in known_builtin_keys),
            None,
        )
    return {
        "id": workflow.id,
        "workflow_id": workflow.id,
        "builtin_key": builtin_key,
        "name": workflow.name,
        "description": workflow.description,
        "visibility": workflow.visibility,
        "is_builtin": workflow.is_builtin,
        "is_default": builtin_key == default_agent_workflow_key(),
        "supports_replans": workflow_supports_replans(spec),
        "supports_long_running_tasks": bool(
            ((spec.get("runtime") or {}).get("features") or {}).get("supports_long_running_tasks")
        ),
        "created_at": iso_utc_z(workflow.created_at) if workflow.created_at else None,
        "updated_at": iso_utc_z(workflow.updated_at) if workflow.updated_at else None,
        **catalog_payload(workflow),
    }


def _definition_for_workflow(workflow) -> AgentDefinition:
    return AgentDefinition(
        definition_id=str(workflow.id),
        framework=str(getattr(workflow, "framework", None) or "langgraph"),
        builder_id=str(getattr(workflow, "builder_id", None) or "langgraph_graph"),
        category=getattr(workflow, "category", None),
        display_name=getattr(workflow, "name", None),
        definition_version=str(getattr(workflow, "version", "")) or None,
    )


def _provider_for_workflow(workflow):
    return builder_for_definition(_definition_for_workflow(workflow))


def _provider_validation_report(provider, spec: Dict[str, Any]) -> Dict[str, Any]:
    report = provider.report(spec) if hasattr(provider, "report") else {}
    return dict(report) if isinstance(report, dict) else {}


def _workflow_spec_payload(workflow) -> Dict[str, Any]:
    try:
        validation = {
            "valid": bool(workflow.validation_result_json.get("valid", True)),
            **(workflow.validation_result_json if isinstance(workflow.validation_result_json, dict) else {}),
        }
    except Exception as exc:
        validation = {
            "valid": False,
            "errors": [f"validation failed: {exc}"],
            "warnings": [],
            "schema_version": getattr(workflow, "schema_version", None),
            "workflow_id": None,
        }
    return {
        "id": str((workflow.metadata_json or {}).get("version_id") or f"{workflow.id}:v{workflow.version}"),
        "workflow_id": workflow.id,
        "framework": getattr(workflow, "framework", "langgraph"),
        "builder_id": getattr(workflow, "builder_id", "langgraph_graph"),
        "category": getattr(workflow, "category", None),
        "version": workflow.version,
        "schema_version": workflow.schema_version,
        "spec_json": workflow.spec_json if isinstance(workflow.spec_json, dict) else {},
        "validation": validation,
        "validation_result_json": workflow.validation_result_json if isinstance(workflow.validation_result_json, dict) else {},
        "created_at": iso_utc_z(workflow.created_at) if workflow.created_at else None,
        "updated_at": iso_utc_z(workflow.updated_at) if workflow.updated_at else None,
    }


def _is_valid_workflow_for_service(workflow) -> bool:
    if not workflow or workflow.schema_version != 2 or not isinstance(workflow.spec_json, dict):
        return False
    validation = workflow.validation_result_json if isinstance(workflow.validation_result_json, dict) else {}
    return bool(validation.get("valid", True))


def _debug_payload_for_response(run) -> Dict[str, Any] | None:
    debug = run.debug_trace_json if isinstance(run.debug_trace_json, dict) else None
    if not debug or debug.get("version") != 1:
        return None
    trace = debug.get("trace") if isinstance(debug.get("trace"), dict) else None
    summary = debug.get("summary") if isinstance(debug.get("summary"), dict) else None
    if trace is None or summary is None:
        return None
    compact_debug = {key: value for key, value in debug.items() if key != "details"}
    return {
        **compact_debug,
        "trace": trace,
        "summary": summary,
        "detail_manifest": detail_manifest(debug.get("details")),
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
        "framework": getattr(run, "framework", "langgraph"),
        "builder_id": getattr(run, "builder_id", "langgraph_graph"),
        "definition_category": getattr(run, "definition_category", None),
        "task_id": run.task_id,
        "parent_run_id": run.parent_run_id,
        "task_attempt": run.task_attempt,
        "turns": [_turn_summary_payload(turn) for turn in turns],
        "resolved_spec_json": run.resolved_spec_json,
        "status": run.status,
        "checkpoint_thread_id": run.checkpoint_thread_id,
        "runtime_binding_version": getattr(run, "runtime_binding_version", 1),
        "runtime_binding_status": getattr(run, "runtime_binding_status", "active"),
        "pending_interrupt": _pending_interrupt_payload(run),
        "started_at": iso_utc_z(run.started_at) if run.started_at else None,
        "completed_at": iso_utc_z(run.completed_at) if run.completed_at else None,
        "error_json": run.error_json,
        "metrics_json": run.metrics_json,
        "parallel_summary": (run.metrics_json or {}).get("parallel_summary") if isinstance(run.metrics_json, dict) else None,
        "corrective": (run.metrics_json or {}).get("corrective") if isinstance(run.metrics_json, dict) else None,
        "retrieval_quality_report": (run.metrics_json or {}).get("retrieval_quality_report") if isinstance(run.metrics_json, dict) else None,
        "grounding_report": (run.metrics_json or {}).get("grounding_report") if isinstance(run.metrics_json, dict) else None,
        "debug": _debug_payload_for_response(run),
        "run_kind": (run.run_metadata_json or {}).get("run_kind"),
        "builder_session_id": (run.run_metadata_json or {}).get("builder_session_id"),
        "final_output": (run.debug_trace_json or {}).get("final_output") if isinstance(run.debug_trace_json, dict) else None,
    }
    return payload


def _sse(event: Dict[str, Any], sequence: int) -> str:
    name = str(event.get("event") or "message")
    payload = {"id": sequence, "event": name, "data": event.get("data") or {}}
    return f"id: {sequence}\nevent: {name}\ndata: {json.dumps(payload, default=str)}\n\n"


def _run_summary_payload(run) -> Dict[str, Any]:
    metrics = run.metrics_json if isinstance(run.metrics_json, dict) else {}
    error = run.error_json if isinstance(run.error_json, dict) else None
    return {
        "id": run.id,
        "thread_id": run.thread_id,
        "workflow_id": run.workflow_id,
        "task_id": run.task_id,
        "parent_run_id": run.parent_run_id,
        "task_attempt": run.task_attempt,
        "status": run.status,
        "pending_interrupt": _pending_interrupt_payload(run),
        "started_at": iso_utc_z(run.started_at) if run.started_at else None,
        "completed_at": iso_utc_z(run.completed_at) if run.completed_at else None,
        "parallel_summary": metrics.get("parallel_summary") if isinstance(metrics.get("parallel_summary"), dict) else None,
        "corrective": metrics.get("corrective") if isinstance(metrics.get("corrective"), dict) else None,
        "retrieval_quality_report": metrics.get("retrieval_quality_report") if isinstance(metrics.get("retrieval_quality_report"), dict) else None,
        "grounding_report": metrics.get("grounding_report") if isinstance(metrics.get("grounding_report"), dict) else None,
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
            "corrective": metrics.get("corrective") if isinstance(metrics.get("corrective"), dict) else None,
        },
        "error": {
            "code": error.get("code"),
            "raw_message": error.get("raw_message"),
            "retryable": error.get("retryable"),
        } if error else None,
    }


def _capabilities_for_workflow(spec_json: Dict[str, Any]) -> Dict[str, Any]:
    runtime = spec_json.get("runtime") if isinstance(spec_json.get("runtime"), dict) else {}
    features = runtime.get("features") if isinstance(runtime.get("features"), dict) else {}
    config = spec_json.get("config") if isinstance(spec_json.get("config"), dict) else {}
    return {
        "required_tool_ids": sorted(workflow_required_tool_ids(spec_json)),
        "node_tool_requirements": dict(sorted(workflow_node_tool_requirements(spec_json).items())),
        "supports_parallel_dispatch": bool(features.get("supports_parallel_dispatch")),
        "supports_corrective_retrieval": bool(features.get("supports_corrective_retrieval")),
        "parallel_policy": config.get("parallel_policy") if isinstance(config.get("parallel_policy"), dict) else None,
        "corrective_policy": config.get("corrective_policy") if isinstance(config.get("corrective_policy"), dict) else None,
    }


def _agent_workflow_tool_contract_catalog(*, excluded_node_types: set[str] | None = None) -> Dict[str, Any]:
    excluded_node_types = excluded_node_types or set()
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
                    if node_type and str(node_type) not in excluded_node_types
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
    repo = AgentWorkflowRepository()
    await repo.seed_builtin_workflows()
    workflows = await repo.list_workflows(include_custom=True)
    valid_workflows = []
    for workflow in workflows:
        try:
            if _is_valid_workflow_for_service(workflow):
                valid_workflows.append(workflow)
        except Exception:
            continue
    return {"agent_workflows": [_workflow_payload(workflow) for workflow in valid_workflows]}


@router.get("/agent-workflows/builtins/{builtin_key}/source")
async def get_builtin_agent_workflow_source(builtin_key: str):
    """Return the immutable-on-disk definition used to seed a built-in workflow."""
    requested_key = builtin_key
    workflow = next(
        (item for item in load_builtin_workflows() if item.get("builtin_key") == builtin_key),
        None,
    )
    if workflow is None:
        stored_workflow = await AgentWorkflowRepository().get_workflow(requested_key)
        if stored_workflow is not None:
            canonical_key = _workflow_payload(stored_workflow).get("builtin_key")
            workflow = next(
                (item for item in load_builtin_workflows() if item.get("builtin_key") == canonical_key),
                None,
            )
            builtin_key = canonical_key or requested_key
    if workflow is None:
        raise HTTPException(status_code=404, detail="Built-in agent workflow source not found")
    definition = AgentDefinition(
        definition_id=builtin_key,
        framework=str(workflow.get("framework") or "langgraph"),
        builder_id=str(workflow.get("builder_id") or "langgraph_graph"),
    )
    try:
        return dict(await builder_for_definition(definition).source(builtin_key))
    except (BuilderSelectionError, KeyError) as exc:
        raise HTTPException(status_code=404, detail="Built-in agent workflow source not found") from exc


@router.post("/agent-workflows/validate")
async def validate_agent_workflow(req: WorkflowValidationRequest):
    definition = AgentDefinition(
        definition_id=str(req.spec.get("workflow_id") or "validation"),
        framework=req.framework,
        builder_id=req.builder_id,
    )
    try:
        provider = builder_for_definition(definition)
        result = await provider.validate(definition, req.spec)
    except BuilderSelectionError as exc:
        raise HTTPException(status_code=400, detail={"code": "builder_unavailable", "message": str(exc)}) from exc
    report = _provider_validation_report(provider, req.spec)
    report.setdefault("framework", req.framework)
    report.setdefault("builder_id", req.builder_id)
    return report


@router.post("/internal/agent-workflows/test-runs/stream")
async def stream_internal_agent_workflow_test(req: BuilderTestRunRequest):
    embedding_context = await _require_ready_thread(req.thread_id)
    thread = embedding_context.thread
    workflow = await AgentWorkflowRepository().get_workflow(req.base_workflow_id, include_custom=True)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Base agent workflow not found")
    if req.use_web_search and not req.allow_external_tools:
        raise HTTPException(
            status_code=409,
            detail={"code": "external_tool_confirmation_required", "message": "Confirm external tool calls before testing with web search."},
        )
    try:
        candidate = dict(req.spec)
        provider = _provider_for_workflow(workflow)
        definition = _definition_for_workflow(workflow)
        resolved = dict(await provider.resolve(
            definition,
            candidate,
            thread_settings={"hitl_web_approval": req.hitl_web_approval},
        ))
    except (BuilderSelectionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail={"code": "invalid_test_workflow", "message": str(exc)}) from exc

    repo = AgentWorkflowRepository()
    run = await repo.create_run(
        thread_id=req.thread_id,
        workflow_id=workflow.id,
        resolved_spec_json=resolved,
        run_metadata_json={
            "run_kind": BUILDER_TEST_RUN_KIND,
            "builder_session_id": req.builder_session_id,
            "base_workflow_id": req.base_workflow_id,
            "spec_fingerprint": spec_fingerprint(resolved),
            "client_spec_fingerprint": req.workflow_spec_fingerprint,
        },
    )

    async def events():
        sequence = 0
        async with open_agent_checkpointer() as checkpointer:
            previous_checkpoint_ids = await delete_previous_builder_tests(req.builder_session_id, keep_run_id=run.id)
            if previous_checkpoint_ids:
                try:
                    await delete_agent_checkpoints(previous_checkpoint_ids, checkpointer=checkpointer)
                except Exception:
                    pass
            try:
                async for event in stream_builder_test(
                    run=run,
                    request=req,
                    embedding_model=embedding_context.embedding_model,
                    checkpointer=checkpointer,
                ):
                    sequence += 1
                    yield _sse(event, sequence)
            finally:
                stored_run = await AgentWorkflowRepository().get_run(run.id)
                if stored_run is not None and stored_run.status != AgentRunStatus.AWAITING_HUMAN.value:
                    try:
                        await delete_agent_checkpoints([str(stored_run.checkpoint_thread_id or stored_run.id)], checkpointer=checkpointer)
                    except Exception:
                        pass
                latest = await latest_builder_test(req.builder_session_id)
                if latest is not None and latest.id != run.id:
                    stale_checkpoint_ids = await delete_previous_builder_tests(req.builder_session_id, keep_run_id=latest.id)
                    if stale_checkpoint_ids:
                        try:
                            await delete_agent_checkpoints(stale_checkpoint_ids, checkpointer=checkpointer)
                        except Exception:
                            pass

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/internal/agent-workflows/test-runs/latest")
async def get_latest_internal_agent_workflow_test(
    builder_session_id: str = Query(..., min_length=1),
    base_workflow_id: Optional[str] = Query(None),
):
    run = await latest_builder_test(builder_session_id, base_workflow_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Builder test run not found")
    turns = await AgentWorkflowRepository().list_chat_turns_for_run(run.id)
    payload = _run_payload(run, turns)
    payload["runtime_inspection"] = await AgentRunService().inspect_agent_run(run)
    return {"agent_run": payload}


@router.post("/internal/agent-workflows/test-runs/{run_id}/cancel")
async def cancel_internal_agent_workflow_test(run_id: str):
    run = await request_builder_test_cancel(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Builder test run not found")
    return {"status": "cancel_requested", "run_id": run.id}


@router.post("/internal/agent-workflows/test-runs/{run_id}/resume/stream")
async def resume_internal_agent_workflow_test(run_id: str, req: BuilderTestRunResumeRequest):
    embedding_context = await _require_ready_thread(req.thread_id)
    thread = embedding_context.thread
    repo = AgentWorkflowRepository()
    run = await repo.get_run(run_id)
    if run is None or run.thread_id != req.thread_id or (run.run_metadata_json or {}).get("run_kind") != BUILDER_TEST_RUN_KIND:
        raise HTTPException(status_code=404, detail="Builder test run not found")
    try:
        resolution = await repo.resolve_pending_interrupt(
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
        raise HTTPException(status_code=exc.http_status, detail={"code": exc.code, "message": str(exc)}) from exc
    if resolution is None:
        raise HTTPException(status_code=404, detail="Builder test run not found")
    decision = (resolution.interrupt or {}).get("decision") if isinstance(resolution.interrupt, dict) else None
    if not isinstance(decision, dict):
        raise HTTPException(status_code=409, detail="Builder test interrupt cannot be resumed")

    async def events():
        sequence = 0
        async with open_agent_checkpointer() as checkpointer:
            try:
                async for event in stream_builder_test(
                    run=resolution.run,
                    request=req,
                    embedding_model=embedding_context.embedding_model,
                    checkpointer=checkpointer,
                    resume_decision=decision,
                ):
                    sequence += 1
                    yield _sse(event, sequence)
            finally:
                stored_run = await AgentWorkflowRepository().get_run(run_id)
                if stored_run is not None and stored_run.status != AgentRunStatus.AWAITING_HUMAN.value:
                    try:
                        await delete_agent_checkpoints([str(stored_run.checkpoint_thread_id or stored_run.id)], checkpointer=checkpointer)
                    except Exception:
                        pass

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@router.get("/agent-workflows/{workflow_id}")
async def get_agent_workflow(workflow_id: str):
    repo = AgentWorkflowRepository()
    await repo.seed_builtin_workflows()
    include_custom = workflow_id not in builtin_workflow_keys()
    workflow = await repo.get_workflow(workflow_id, include_custom=include_custom)
    if not workflow or not _is_valid_workflow_for_service(workflow):
        raise HTTPException(status_code=404, detail="Agent workflow not found")
    spec_payload = _workflow_spec_payload(workflow)
    return {
        "agent_workflow": _workflow_payload(workflow),
        "spec": spec_payload,
        "current_version": spec_payload,
        "capabilities": _capabilities_for_workflow(workflow.spec_json if isinstance(workflow.spec_json, dict) else {}),
    }


@router.post("/internal/agent-workflows")
async def save_internal_agent_workflow(req: InternalAgentWorkflowSaveRequest):
    repo = AgentWorkflowRepository()
    try:
        workflow_id = (req.workflow_id or "").strip() or None
        if workflow_id is None:
            workflow_id = f"custom_workflow_{uuid.uuid4().hex[:12]}"
        spec_json = with_default_runtime(dict(req.spec_json))
        spec_json["workflow_id"] = workflow_id
        workflow, version = await repo.save_internal_workflow_version(
            workflow_id=workflow_id,
            name=req.name,
            description=req.description,
            spec_json=spec_json,
            increment_version=False,
        )
    except (BuilderSelectionError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    version_payload = _workflow_spec_payload(workflow)
    return {
        "agent_workflow": _workflow_payload(workflow),
        "spec": version_payload,
        "version": version_payload,
    }


@router.delete("/internal/agent-workflows/{workflow_id}")
async def delete_internal_agent_workflow(workflow_id: str):
    repo = AgentWorkflowRepository()
    try:
        workflow = await repo.mark_custom_workflow_deleted(workflow_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if workflow is None:
        raise HTTPException(status_code=404, detail="Internal agent workflow not found")
    return {
        "status": "deleted",
        "agent_workflow": _workflow_payload(workflow),
    }


@router.get("/internal/agent-workflows/catalog")
async def get_internal_agent_workflow_catalog():
    definition = AgentDefinition(
        definition_id="catalog",
        framework="langgraph",
        builder_id="langgraph_graph",
    )
    try:
        catalog = await builder_for_definition(definition).catalog(definition)
    except BuilderSelectionError as exc:
        raise HTTPException(status_code=503, detail={"code": "builder_unavailable", "message": str(exc)}) from exc
    return dict(catalog.payload)


@router.get("/internal/agent-workflows/{workflow_id}")
async def get_internal_agent_workflow(workflow_id: str):
    repo = AgentWorkflowRepository()
    workflow = await repo.get_workflow(workflow_id, include_custom=True)
    if not workflow or workflow.is_builtin:
        raise HTTPException(status_code=404, detail="Internal agent workflow not found")
    spec_payload = _workflow_spec_payload(workflow)
    return {
        "agent_workflow": _workflow_payload(workflow),
        "spec": spec_payload,
        "current_version": spec_payload,
    }


@router.post("/threads/{thread_id}/agent-config/validate")
async def validate_thread_agent_config(thread_id: str, req: ThreadAgentConfigValidationRequest):
    thread = await get_thread(thread_id)
    if not thread:
        raise HTTPException(status_code=404, detail="Thread not found")

    repo = AgentWorkflowRepository()
    await repo.seed_builtin_workflows()
    thread_settings = await get_thread_settings(thread_id)
    agent_settings = thread_settings.get("agent_workflow") if isinstance(thread_settings, dict) else None
    agent_settings = agent_settings if isinstance(agent_settings, dict) else {}
    workflow_id = agent_settings.get("workflow_id") or default_agent_workflow_key()

    workflow = await repo.get_workflow(
        workflow_id,
        include_custom=workflow_id not in builtin_workflow_keys(),
    )
    if not workflow:
        raise HTTPException(status_code=404, detail="Agent workflow not found")

    provider = _provider_for_workflow(workflow)
    definition = _definition_for_workflow(workflow)
    try:
        resolved_spec = await provider.resolve(
            definition,
            workflow.spec_json,
            thread_settings=thread_settings,
            request_overrides=req.overrides,
        )
    except ValueError as exc:
        candidate = dict(workflow.spec_json or {})
        candidate_config = dict(candidate.get("config") or {})
        for source in (thread_settings or {}, req.overrides or {}):
            for key in ALLOWED_WORKFLOW_CONFIG_KEYS:
                value = source.get(key) if isinstance(source, dict) else None
                if value is not None:
                    candidate_config[key] = value
        candidate["config"] = candidate_config
        report = _provider_validation_report(provider, candidate)
        report["errors"] = report.get("errors") or [str(exc)]
        report["errors"] = report["errors"] or [str(exc)]
        return {
            "valid": False,
            "workflow_id": workflow.id,
            "workflow_version": workflow.version,
            "validation": report,
            "resolved_spec_json": candidate,
        }

    validation = await provider.validate(definition, resolved_spec)
    return {
        "valid": validation.valid,
        "workflow_id": workflow.id,
        "workflow_version": workflow.version,
        "validation": _provider_validation_report(provider, resolved_spec),
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

    repo = AgentWorkflowRepository()
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

    repo = AgentWorkflowRepository()
    run = await repo.get_run(run_id)
    if not run or run.thread_id != thread_id:
        raise HTTPException(status_code=404, detail="Agent run not found")
    turns = await repo.list_chat_turns_for_run(run.id)
    return {"agent_run": _run_payload(run, turns)}


@router.get("/agent-runs/{run_id}/details")
async def get_agent_run_node_details(
    run_id: str,
    node_id: str = Query(..., min_length=1),
    visit_index: int = Query(..., ge=1),
    thread_id: str = Query(..., min_length=1),
):
    run = await AgentWorkflowRepository().get_run(run_id)
    if run is None or run.thread_id != thread_id or not await get_thread(thread_id):
        raise HTTPException(status_code=404, detail="Agent run not found")
    debug = run.debug_trace_json if isinstance(run.debug_trace_json, dict) else {}
    details = debug.get("details") if isinstance(debug.get("details"), list) else []
    for detail in details:
        if not isinstance(detail, dict):
            continue
        try:
            detail_visit = max(1, int(detail.get("visit_index") or 1))
        except (TypeError, ValueError):
            detail_visit = 1
        if str(detail.get("node_id") or "") == node_id and detail_visit == visit_index:
            return {"run_id": run.id, "detail": detail}
    raise HTTPException(status_code=404, detail="Node visit details are unavailable")


@router.post("/agent-runs/{run_id}/cancel")
async def cancel_chat_agent_run(
    run_id: str,
    req: AgentRunCancelRequest,
):
    if not await get_thread(req.thread_id):
        raise HTTPException(status_code=404, detail="Agent run not found")
    result = await request_chat_run_cancel(run_id, thread_id=req.thread_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    if result.status == "missing":
        raise HTTPException(status_code=404, detail="Agent run not found")
    if result.status == CHAT_CANCEL_UNSUPPORTED:
        raise HTTPException(status_code=409, detail="This run uses its own cancellation endpoint")
    if result.status == CHAT_CANCEL_AWAITING_HUMAN:
        raise HTTPException(status_code=409, detail="Use the human-review actions for this paused run")
    return {
        "status": result.status,
        "run_id": result.run_id,
        "run_status": result.run_status,
    }


@router.post("/agent-runs/{run_id}/resume")
async def resume_agent_run(
    run_id: str,
    req: AgentRunResumeRequest,
    accept: Optional[str] = Header(default=None),
):
    await _require_ready_thread(req.thread_id)

    service = AgentRunService()

    async def execute_resume(*, event_sink: Any = None):
        return await service.resume_agent_run(
            run_id,
            interrupt_id=req.interrupt_id,
            action=req.action,
            edited_payload=req.edited_payload,
            client_metadata=req.client_metadata,
            selected_option_ids=req.selected_option_ids,
            resume_token=req.resume_token,
            resume_version=req.resume_version,
            expected_thread_id=req.thread_id,
            execution_event_sink=event_sink,
        )

    if "text/event-stream" in str(accept or "").lower():
        sink = AgentExecutionEventSink(include_details=False)

        async def run_resume() -> None:
            try:
                result = await execute_resume(event_sink=sink)
                if result is None:
                    await sink.queue.put({"event": "__missing__", "data": {}})
                    return
                compact_run = {
                    "id": result.run.id,
                    "thread_id": result.run.thread_id,
                    "workflow_id": result.run.workflow_id,
                    "status": result.run.status,
                    "pending_interrupt": _pending_interrupt_payload(result.run),
                }
                await sink.queue.put({
                    "event": "__result__",
                    "data": {
                        "agent_run": compact_run,
                        "interrupt": result.interrupt,
                        "outcome": result.outcome,
                        "duplicate": result.duplicate,
                    },
                })
            except AgentRunInterruptError as exc:
                await sink.queue.put({"event": "__error__", "data": {"error": {"code": exc.code, "raw_message": str(exc), "retryable": False}}})
            except Exception as exc:
                await sink.queue.put({"event": "__error__", "data": {"error": {"code": "agent_run_resume_failed", "raw_message": str(exc), "retryable": True}}})

        async def events():
            sequence = 0
            task = asyncio.create_task(run_resume())
            retain_background_task(task)
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(sink.queue.get(), timeout=12)
                    except asyncio.TimeoutError:
                        sequence += 1
                        yield _sse({"event": "heartbeat", "data": {"run_id": run_id}}, sequence)
                        continue
                    event = str(item.get("event") or "message")
                    data = item.get("data") or {}
                    if event == "__missing__":
                        sequence += 1
                        yield _sse({"event": "run.failed", "data": {"run_id": run_id, "error": {"code": "agent_run_not_found", "raw_message": "Agent run not found", "retryable": False}}}, sequence)
                        break
                    if event == "__error__":
                        sequence += 1
                        yield _sse({"event": "run.failed", "data": {"run_id": run_id, **data}}, sequence)
                        break
                    if event == "__result__":
                        status = str((data.get("agent_run") or {}).get("status") or "completed")
                        terminal_event = "interrupt.created" if status == AgentRunStatus.AWAITING_HUMAN.value else "run.failed" if status == AgentRunStatus.FAILED.value else "run.completed"
                        sequence += 1
                        yield _sse({"event": terminal_event, "data": {"run_id": run_id, "status": status, "response": data}}, sequence)
                        break
                    sequence += 1
                    yield _sse(item, sequence)
            finally:
                sink.close()

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    try:
        result = await execute_resume()
    except AgentRunInterruptError as exc:
        raise HTTPException(
            status_code=exc.http_status,
            detail={"code": exc.code, "message": str(exc)},
        ) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    repo = AgentWorkflowRepository()
    turns = await repo.list_chat_turns_for_run(result.run.id)
    return {
        "agent_run": _run_payload(result.run, turns),
        "interrupt": result.interrupt,
        "outcome": result.outcome,
        "duplicate": result.duplicate,
    }
