from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import socket
import time
from contextlib import suppress
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Optional

from app.agent_workflows.debug_trace import AgentTraceRecorder, finalize_and_merge_debug_payload
from app.agent_workflows.execution_stream import AgentExecutionEventSink
from app.agent_workflows.repository import AgentWorkflowRepository
from app.db import AgentRunStatus, get_recent_messages, get_thread, get_thread_settings
from app.mcp.execution_context_token import issue_execution_context_token
from app.models.deep_research import AgentTaskStatus
from app.services import agent_task_repository as tasks
from app.services.task_artifact_service import persist_task_artifact
from app.services.agent_task_maintenance import MAINTENANCE_INTERVAL_SECONDS, run_task_maintenance
from app.runtime.adapter import RuntimeExecutionContext
from app.runtime.contracts import AgentRuntimeRequest, AgentRuntimeResult, RuntimeOperationId
from app.runtime.capability_resolver import require_capability
from app.runtime.errors import RuntimeError as AgentRuntimeError
from app.runtime.catalog import (
    continuation_from_run,
    definition_from_run,
    definition_from_workflow,
    result_to_product_payload,
)
from app.runtime.registry import adapter_for_definition, get_runtime_registry
from app.runtime.builder_registry import builder_for_definition
from app.runtime.hermes_config import hermes_model_context_length
from app.runtime.budgets import deep_agent_budgets
from app.tools.context import ToolInvocationContext


logger = logging.getLogger(__name__)
LEASE_SECONDS = 60
HEARTBEAT_SECONDS = 15
HERMES_DOCUMENT_EVIDENCE_TOOLS = frozenset({"search_documents", "search_document_by_id"})
HERMES_RESEARCH_EVIDENCE_TOOLS = frozenset({
    *HERMES_DOCUMENT_EVIDENCE_TOOLS,
    "search_durable_memory", "search_web", "wikipedia", "wikidata", "arxiv",
    "pubmed", "semantic_scholar", "stack_exchange", "yahoo_finance_news",
})


async def _invoke_task_runtime(
    *,
    adapter: Any,
    definition: Any,
    run: Any,
    runtime_request: AgentRuntimeRequest,
    runtime_context: RuntimeExecutionContext,
    runtime_event_sink: Any,
    repository: AgentWorkflowRepository,
) -> AgentRuntimeResult | None:
    """Dispatch one task attempt using its explicit lifecycle contract."""

    pending = dict(run.pending_interrupt_json or {})
    if getattr(run, "_fresh_runtime_run", False):
        await require_capability(
            definition,
            RuntimeOperationId.RUN_START,
            registry=get_runtime_registry(),
            run=run,
        )
        result = await adapter.start(
            runtime_request,
            context=runtime_context,
            event_sink=runtime_event_sink,
        )
        await repository.mark_runtime_started(run.id)
        return result

    if pending.get("status") in {"resumed", "resolved"} and isinstance(pending.get("decision"), dict):
        response_operation = pending.get("response_operation")
        if response_operation == RuntimeOperationId.RUN_RESUME.value:
            return await adapter.resume(
                runtime_request,
                interrupt=pending,
                context=runtime_context,
                event_sink=runtime_event_sink,
            )
        if response_operation == RuntimeOperationId.RUN_APPROVAL_RESPOND.value:
            return await adapter.continue_run(
                runtime_request,
                context=runtime_context,
                event_sink=runtime_event_sink,
            )
        raise AgentRuntimeError(
            code="interrupt_response_operation_invalid",
            safe_message="The pending interrupt does not declare a supported response operation",
            retryable=False,
            details={"response_operation": response_operation},
        )

    return await adapter.continue_run(
        runtime_request,
        context=runtime_context,
        event_sink=runtime_event_sink,
    )


def _hermes_grounding_summary(events: list[Any], *, documents_present: bool) -> dict[str, Any]:
    successful: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for event in events:
        payload = dict(getattr(event, "payload_json", None) or getattr(event, "payload", None) or {})
        if payload.get("source") != "askpdf_mcp":
            continue
        name = str(payload.get("tool_name") or "")
        if getattr(event, "kind", "") == "tool.completed" and payload.get("ok") is True and int(payload.get("result_count") or 0) > 0:
            successful.append(payload)
        elif getattr(event, "kind", "") == "tool.failed":
            failures.append(payload)
    eligible = HERMES_DOCUMENT_EVIDENCE_TOOLS if documents_present else HERMES_RESEARCH_EVIDENCE_TOOLS
    qualifying = [item for item in successful if item.get("tool_name") in eligible]
    return {
        "required": True,
        "requirement": "document" if documents_present else "research",
        "grounded": bool(qualifying),
        "evidence_result_count": sum(int(item.get("result_count") or 0) for item in qualifying),
        "successful_evidence_tools": sorted({str(item.get("tool_name")) for item in qualifying}),
        "failed_tool_count": len(failures),
        "failure_codes": sorted({str((item.get("error") or {}).get("code") or "tool_failed") for item in failures}),
    }


async def _task_context_snapshot(task: Any, thread: Any, config: dict[str, Any]) -> dict[str, Any]:
    """Create a bounded, deterministic context seed; retrieval remains MCP-backed."""

    messages = await get_recent_messages(task.thread_id, limit=20)
    conversation: list[dict[str, str]] = []
    context_window = int(config.get("context_window") or hermes_model_context_length(required=True))
    remaining = min(24_000, max(4_000, context_window))
    for message in reversed(messages):
        content = str(getattr(message, "context_compact", None) or getattr(message, "content", "")).strip()
        if not content or remaining <= 0:
            continue
        content = content[-remaining:]
        conversation.append({"role": str(getattr(message, "role", "user")), "content": content})
        remaining -= len(content)
    conversation.reverse()
    documents = []
    for file_hash, metadata in sorted(dict(getattr(thread, "documents_meta", None) or {}).items()):
        item = dict(metadata or {}) if isinstance(metadata, dict) else {}
        documents.append({
            "file_hash": str(file_hash),
            "name": str(item.get("file_name") or item.get("filename") or file_hash),
        })
    return {
        "objective": task.objective,
        "thread_id": task.thread_id,
        "project_id": task.project_id,
        "model": config.get("llm_model"),
        "embedding_model": thread.embedding_model,
        "context_window": context_window,
        "limits": dict(config.get("limits") or {}),
        "recent_conversation": conversation,
        "documents": documents,
    }


async def _complete_run_with_trace(
    repository: AgentWorkflowRepository,
    *,
    run: Any,
    recorder: AgentTraceRecorder,
    status: str,
    metrics: dict[str, Any],
    result: dict[str, Any],
    error: Optional[dict[str, Any]] = None,
) -> Any:
    """Atomically persist one terminal AgentRun and its merged trace payload."""

    completed_at = datetime.now(timezone.utc)
    debug_payload = finalize_and_merge_debug_payload(
        recorder=recorder,
        run=run,
        metrics=metrics,
        result=result,
        route=result.get("route"),
        route_reason=result.get("route_reason"),
        error=error,
        run_status=status,
        completed_at=completed_at,
    )
    return await repository.complete_run(
        run.id,
        status=status,
        metrics_json=metrics,
        error_json=error,
        debug_trace_json=debug_payload,
        completed_at=completed_at,
    )


async def ensure_task_run(task_id: str):
    task = await tasks.get_task(task_id)
    if task is None:
        raise ValueError("task_not_found")
    active = await tasks.get_task_run(task_id)
    if active is not None and active.status in {AgentRunStatus.RUNNING.value, AgentRunStatus.AWAITING_HUMAN.value}:
        # An existing active run may have a checkpoint to continue after a
        # worker restart.  Mark it explicitly so a newly-created run is not
        # mistaken for a continuation merely because LangGraph reserves its
        # checkpoint thread ID at creation time.
        metadata = dict(active.run_metadata_json or {})
        setattr(active, "_fresh_runtime_run", metadata.get("runtime_started") is False)
        return active

    repository = AgentWorkflowRepository()
    workflow = await repository.get_workflow(task.workflow_id, include_custom=False)
    if workflow is None:
        await repository.seed_builtin_workflows()
        workflow = await repository.get_workflow(task.workflow_id, include_custom=False)
    if workflow is None:
        raise RuntimeError("deep_research_workflow_unavailable")

    thread_settings = await get_thread_settings(task.thread_id)
    definition = definition_from_workflow(workflow)
    provider = builder_for_definition(definition)
    resolved = await provider.resolve(
        definition,
        workflow.spec_json,
        thread_settings=thread_settings,
        request_overrides={
            "llm_model": (task.config_json or {}).get("llm_model"),
            "context_window": (task.config_json or {}).get("context_window"),
            "use_web_search": bool((task.config_json or {}).get("use_web_search")),
        },
    )
    config = dict(resolved.get("config") or {})
    task_policy = dict(config.get("task_policy") or {})
    task_policy["limits"] = dict((task.config_json or {}).get("limits") or {})
    task_policy["profiles"] = list((task.config_json or {}).get("enabled_profiles") or [])
    config["task_policy"] = task_policy
    config["use_web_search"] = bool((task.config_json or {}).get("use_web_search"))
    resolved["config"] = config
    frozen_spec = dict(await provider.normalize(definition, resolved))
    metadata = dict(getattr(workflow, "metadata_json", None) or {})
    version = int(metadata.get("version") or workflow.schema_version or 1)
    run = await repository.create_run(
        thread_id=task.thread_id,
        workflow_id=workflow.id,
        workflow_version_id=str(metadata.get("version_id") or f"{workflow.id}:v{version}"),
        workflow_version=version,
        framework=definition.framework,
        builder_id=definition.builder_id,
        definition_category=getattr(workflow, "category", None),
        resolved_spec_json=frozen_spec,
        user_id=task.user_id,
        run_metadata_json={
            "executed_workflow_id": workflow.id,
            "run_kind": "agent_task",
            "agent_task_id": task.id,
            "runtime_started": False,
        },
    )
    # create_run intentionally allocates the LangGraph checkpoint identity up
    # front.  That identity is not evidence that a checkpoint exists yet.
    # Keep this process-local marker until the first start operation completes.
    setattr(run, "_fresh_runtime_run", True)
    return await tasks.attach_run(task.id, run, parent_run_id=active.id if active is not None else None)


async def _heartbeat(task_id: str, worker_id: str) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        if not await tasks.heartbeat_task(task_id, worker_id, lease_seconds=LEASE_SECONDS):
            return


async def execute_claimed_task(task_id: str, worker_id: str) -> None:
    task = await tasks.get_task(task_id)
    if task is None:
        return
    if task.status == AgentTaskStatus.CANCELLING.value:
        active_run = await tasks.get_task_run(task_id)
        if active_run is not None and active_run.status in {
            AgentRunStatus.RUNNING.value,
            AgentRunStatus.AWAITING_HUMAN.value,
        }:
            await AgentWorkflowRepository().complete_run(
                active_run.id,
                status=AgentRunStatus.CANCELLED.value,
                error_json={"code": "agent_task_cancelled", "retryable": False},
            )
        await tasks.complete_task(
            task_id,
            status=AgentTaskStatus.CANCELLED.value,
            reason="cancelled_by_user",
        )
        await tasks.release_task_lease(task_id, worker_id, lease_seconds=LEASE_SECONDS)
        return
    if await tasks.active_runtime_budget_exhausted(task_id):
        await tasks.complete_task(
            task_id,
            status=AgentTaskStatus.FAILED.value,
            reason="active_runtime_budget_exhausted",
        )
        await tasks.release_task_lease(task_id, worker_id, lease_seconds=LEASE_SECONDS)
        return
    run = await ensure_task_run(task_id)
    task = await tasks.get_task(task_id)
    thread = await get_thread(task.thread_id) if task else None
    if task is None or thread is None:
        await tasks.complete_task(task_id, status=AgentTaskStatus.FAILED.value, reason="task_thread_missing")
        await tasks.release_task_lease(task_id, worker_id, lease_seconds=LEASE_SECONDS)
        return

    config = dict(task.config_json or {})
    existing_artifacts = await tasks.list_artifacts(task.id)
    artifact_manifest: list[dict[str, Any]] = []
    artifact_contents: dict[str, str] = {}
    from app.services.content_store import get_content_store
    content_store = get_content_store()
    for artifact in existing_artifacts[: int(config.get("max_context_artifacts", 200))]:
        manifest = {
            "id": artifact.id, "kind": artifact.kind, "sha256": artifact.sha256,
            "byte_size": artifact.byte_size, "summary": artifact.summary_json,
            "todo_id": artifact.todo_id, "subagent_run_id": artifact.subagent_run_id,
            "provenance": dict(artifact.provenance_json or {}),
        }
        artifact_manifest.append(manifest)
        if artifact.kind in {"intermediate_report", "context_summary", "tool_output"} and artifact.byte_size <= 20_000:
            try:
                artifact_contents[artifact.id] = (await content_store.read(artifact.object_key)).decode("utf-8", errors="replace")
            except (FileNotFoundError, OSError):
                continue
    # Some task runs were created before the run snapshot was populated. Do
    # not create a replacement run: reconstruct the same concrete deep
    # workflow definition and attach its materialized spec to this run before
    # continuation is sent to the runtime.
    if not isinstance(run.resolved_spec_json, dict) or not run.resolved_spec_json.get("config"):
        workflow = await AgentWorkflowRepository().get_workflow(task.workflow_id, include_custom=False)
        if workflow is None:
            await AgentWorkflowRepository().seed_builtin_workflows()
            workflow = await AgentWorkflowRepository().get_workflow(task.workflow_id, include_custom=False)
        if workflow is None:
            raise RuntimeError("deep_research_workflow_unavailable")
        thread_settings = await get_thread_settings(task.thread_id)
        definition = definition_from_workflow(workflow)
        provider = builder_for_definition(definition)
        resolved = await provider.resolve(
            definition,
            workflow.spec_json,
            thread_settings=thread_settings,
            request_overrides={
                "llm_model": config.get("llm_model"),
                "context_window": config.get("context_window"),
                "use_web_search": bool(config.get("use_web_search")),
            },
        )
        resolved_config = dict(resolved.get("config") or {})
        task_policy = dict(resolved_config.get("task_policy") or {})
        task_policy["limits"] = dict(config.get("limits") or {})
        task_policy["profiles"] = list(config.get("enabled_profiles") or [])
        resolved_config["task_policy"] = task_policy
        resolved_config["use_web_search"] = bool(config.get("use_web_search"))
        resolved["config"] = resolved_config
        run.resolved_spec_json = dict(await provider.normalize(definition, resolved))
    todos = await tasks.list_todos(task.id)
    task_web_access = await tasks.get_task_web_access(task.id)
    request = SimpleNamespace(
        question=task.objective,
        llm_model=config.get("llm_model"),
        context_window=config.get("context_window"),
        use_web_search=bool(config.get("use_web_search")),
        web_search_mode=str(config.get("web_search_mode") or "off"),
        task_web_access=task_web_access,
        use_reranker=True,
        bypass_clarification=True,
        system_role_override="",
        tool_instructions_override={},
        custom_instructions_override="",
        client_timezone=None,
        client_locale=None,
        client_now_iso=None,
        agent_task_id=task.id,
        agent_task_version=task.version,
        task_enabled_profiles=list(config.get("enabled_profiles") or []),
        task_limits=dict(config.get("limits") or {}),
        task_plan_revision=max((todo.updated_revision for todo in todos), default=0),
        task_run_plan_count=0,
        task_todos=[{
            "id": todo.id, "title": todo.title, "description": todo.description,
            "completion_criteria": todo.completion_criteria, "status": todo.status,
            "priority": todo.priority, "required": todo.required,
            "dependency_ids": list(todo.dependency_ids_json or []), "profile_id": todo.profile_id,
            "attempt": todo.attempt, "max_attempts": todo.max_attempts,
            "progress": todo.progress, "result_summary": todo.result_summary,
            "artifact_ids": list(todo.artifact_ids_json or []), "version": todo.version,
        } for todo in todos],
        task_budget_usage=dict(task.budgets_json or {}),
        runtime_execution_mode=True,
        runtime_artifact_manifest=artifact_manifest,
        runtime_artifact_contents=artifact_contents,
    )
    repository = AgentWorkflowRepository()
    trace = AgentTraceRecorder(run)
    context = {
        "agent_run_id": run.id,
        "agent_workflow_id": run.workflow_id,
        "agent_workflow_version": run.workflow_version,
        "checkpoint_thread_id": run.checkpoint_thread_id,
    }
    heartbeat = asyncio.create_task(_heartbeat(task.id, worker_id))
    started = time.perf_counter()
    async def cancellation_requested() -> bool:
        return await tasks.task_cancel_requested(task.id) or await tasks.active_runtime_budget_exhausted(task.id)

    try:
        definition = definition_from_run(run)
        adapter = adapter_for_definition(definition)
        # Task runs created before the external runtime was introduced may
        # contain a materialized graph without the v2 envelope marker.  Keep
        # the same run/continuation identity and normalize only the in-flight
        # snapshot sent to the runtime so those paused runs remain resumable.
        resolved_spec = dict(await builder_for_definition(definition).normalize(
            definition, dict(run.resolved_spec_json or {})
        ))
        if resolved_spec != (run.resolved_spec_json or {}):
            run.resolved_spec_json = resolved_spec
        runtime_input: dict[str, Any] = {"question": task.objective}
        if definition.framework == "hermes":
            snapshot = await _task_context_snapshot(task, thread, config)
            allowed_tools = list(
                ((resolved_spec.get("managed_profile") or {}).get("mcp") or {}).get("allowed_tool_ids")
                or (resolved_spec.get("config") or {}).get("allowed_tool_ids")
                or []
            )
            token_ttl_seconds = max(3600, int((config.get("limits") or {}).get("max_active_runtime_ms", 3_600_000)) // 1000)
            token = issue_execution_context_token(
                ToolInvocationContext(
                    thread_id=task.thread_id,
                    run_id=run.id,
                    embedding_model=thread.embedding_model,
                    context_window=int(config.get("context_window") or hermes_model_context_length(required=True)),
                    use_web_search=bool(config.get("use_web_search")),
                    use_reranker=True,
                    extensions={
                        "task_id": task.id,
                        "llm_model": config.get("llm_model"),
                        "web_search_mode": config.get("web_search_mode", "off"),
                    },
                ),
                task_id=task.id,
                allowed_tools=allowed_tools,
                ttl_seconds=token_ttl_seconds,
            )
            await repository.update_run_metadata_fields(run.id, {
                "hermes_mcp_context": {
                    "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                    "expires_at_epoch": int(time.time()) + token_ttl_seconds,
                    "context_window": int(config.get("context_window") or hermes_model_context_length(required=True)),
                },
            })
            runtime_input.update({"task_context": snapshot, "mcp_execution_context_token": token})
        runtime_request = AgentRuntimeRequest(
            run_id=run.id,
            thread_id=run.thread_id,
            definition_id=definition.definition_id,
            framework=definition.framework,
            builder_id=definition.builder_id,
            input=runtime_input,
            task_id=task.id,
            continuation=continuation_from_run(run),
        )
        runtime_context = RuntimeExecutionContext(
            request=request,
            embedding_model=thread.embedding_model,
            resolved_spec=resolved_spec,
            agent_run_context={**context, "run": run},
            trace_recorder=trace,
            cancellation_checker=cancellation_requested,
            task_id=task.id,
            task_worker_id=worker_id,
        )
        runtime_event_sink = None
        if definition.framework == "hermes":
            runtime_event_sink = AgentExecutionEventSink(include_details=False)
            runtime_event_sink.bind_trace_recorder(trace)
            runtime_event_sink.bind_runtime_binding_persister(repository.update_runtime_binding)
            runtime_event_sink.bind_runtime_event_persister(run.id, repository.append_run_event)
        runtime_result = await _invoke_task_runtime(
            adapter=adapter,
            definition=definition,
            run=run,
            runtime_request=runtime_request,
            runtime_context=runtime_context,
            runtime_event_sink=runtime_event_sink,
            repository=repository,
        )
        if runtime_result is None:
            # A continuation is optional at the runtime boundary. A missing
            # checkpoint is a terminal runtime outcome.
            runtime_result = AgentRuntimeResult(
                status="failed",
                error={
                    "code": "runtime_continuation_missing",
                    "message": "The LangGraph run has no durable checkpoint to continue",
                    "retryable": False,
                },
            )
        if runtime_result.continuation is not None:
            await repository.update_runtime_binding(run.id, runtime_result.continuation)
        result = result_to_product_payload(runtime_result)
        # Runtime artifacts are data, not product records. Project them in
        # rag-service after the stream completes and translate deterministic
        # runtime IDs to the persisted artifact IDs used by task APIs.
        runtime_artifacts = [value for value in result.get("runtime_artifacts") or [] if isinstance(value, dict) and value.get("content") is not None]
        artifact_id_map: dict[str, str] = {}
        for artifact in runtime_artifacts:
            projected = await persist_task_artifact(
                task_id=task.id,
                agent_run_id=run.id,
                kind=str(artifact.get("kind") or "tool_output"),
                content=str(artifact.get("content") or ""),
                media_type=str(artifact.get("media_type") or "text/plain"),
                todo_id=artifact.get("todo_id"),
                # Runtime subagent IDs are intentionally opaque and are not
                # product FK values; task ownership is retained in provenance.
                provenance={**dict(artifact.get("provenance") or {}), "runtime_artifact_id": artifact.get("artifact_id"), "runtime_subagent_run_id": artifact.get("subagent_run_id")},
                source_refs=dict(artifact.get("source_refs") or {}),
            )
            if artifact.get("artifact_id"):
                artifact_id_map[str(artifact["artifact_id"])] = projected.id
        if artifact_id_map:
            def replace_ids(value: Any) -> Any:
                if isinstance(value, str):
                    return artifact_id_map.get(value, value)
                if isinstance(value, list):
                    return [replace_ids(item) for item in value]
                if isinstance(value, dict):
                    return {key: replace_ids(item) for key, item in value.items()}
                return value
            for key in ("task_todos", "task_artifact_manifest", "task_evidence_manifest", "task_result_packets"):
                if key in result:
                    result[key] = replace_ids(result[key])
        status = str(result.get("status") or AgentRunStatus.COMPLETED.value)
        metrics = dict(run.metrics_json or {})
        metrics.update({"duration_ms": round((time.perf_counter() - started) * 1000, 2)})
        if status == AgentRunStatus.AWAITING_HUMAN.value:
            pending = dict(result.get("pending_interrupt") or {})
            trace.record_interrupted_snapshot(interrupt=pending, state=result)
            trace.record_runtime_event(
                "checkpoint.created",
                attributes={
                    "askpdf.run.id": run.id,
                    "askpdf.thread.id": task.thread_id,
                    "askpdf.checkpoint.thread_id": run.checkpoint_thread_id,
                    "askpdf.status": AgentRunStatus.AWAITING_HUMAN.value,
                },
                output_data={
                    "interrupt_id": pending.get("interrupt_id"),
                    "route": result.get("route"),
                },
            )
            debug_payload = finalize_and_merge_debug_payload(
                recorder=trace,
                run=run,
                metrics=metrics,
                result=result,
                route=result.get("route"),
                route_reason=result.get("route_reason"),
                run_status=AgentRunStatus.AWAITING_HUMAN.value,
            )
            await repository.mark_run_awaiting_human(
                run.id,
                pending,
                metrics_json=metrics,
                debug_trace_json=debug_payload,
            )
            task_status = AgentTaskStatus.PAUSED.value if pending.get("type") == "task_pause" else AgentTaskStatus.AWAITING_APPROVAL.value
            await tasks.set_task_runtime_status(task.id, task_status, phase="checkpointed_interrupt")
            if task_status == AgentTaskStatus.AWAITING_APPROVAL.value:
                await tasks.append_event(
                    task.id,
                    "task.approval_requested",
                    agent_run_id=run.id,
                    payload={
                        "interrupt_id": pending.get("interrupt_id"),
                        "title": pending.get("title"),
                        "type": pending.get("type"),
                        "approval_scope_kind": pending.get("approval_scope_kind"),
                    },
                )
            return
        if status == AgentRunStatus.CANCELLED.value:
            latest_task = await tasks.get_task(task.id)
            budget_exhausted = bool(latest_task and latest_task.terminal_reason == "active_runtime_budget_exhausted")
            await _complete_run_with_trace(
                repository,
                run=run,
                recorder=trace,
                status=status,
                metrics=metrics,
                result=result,
            )
            await tasks.complete_task(
                task.id,
                status=AgentTaskStatus.FAILED.value if budget_exhausted else AgentTaskStatus.CANCELLED.value,
                reason="active_runtime_budget_exhausted" if budget_exhausted else "cancelled_by_user",
            )
            return
        error = result.get("agent_error") if isinstance(result.get("agent_error"), dict) else None
        if status == AgentRunStatus.COMPLETED.value:
            terminal_todos = await tasks.list_todos(task.id)
            incomplete = any(todo.required and todo.status != "completed" for todo in terminal_todos)
            final_answer = str(result.get("final_answer") or result.get("answer") or "").strip()
            if not final_answer:
                terminal_error = {"code": "final_report_missing", "retryable": True}
                await _complete_run_with_trace(
                    repository,
                    run=run,
                    recorder=trace,
                    status=AgentRunStatus.FAILED.value,
                    metrics=metrics,
                    result=result,
                    error=terminal_error,
                )
                await tasks.complete_task(task.id, status=AgentTaskStatus.FAILED.value, reason="final_report_missing")
                return
            evidence_policy = dict((resolved_spec.get("config") or {}).get("task_policy") or {}).get("evidence")
            if definition.framework == "hermes" and evidence_policy == "document_when_available":
                grounding = _hermes_grounding_summary(
                    await repository.list_run_events(run.id),
                    documents_present=bool(dict(getattr(thread, "documents_meta", None) or {})),
                )
                metrics["grounding"] = grounding
                result["grounding"] = grounding
                if not grounding["grounded"]:
                    terminal_error = {
                        "code": "required_evidence_unavailable",
                        "message": "Hermes did not return the evidence required for this research task",
                        "retryable": True,
                        "details": grounding,
                    }
                    await _complete_run_with_trace(
                        repository, run=run, recorder=trace,
                        status=AgentRunStatus.FAILED.value, metrics=metrics,
                        result=result, error=terminal_error,
                    )
                    await tasks.complete_task(
                        task.id, status=AgentTaskStatus.FAILED.value,
                        reason="required_evidence_unavailable",
                    )
                    return
            evidence_manifest = [
                value for value in result.get("task_evidence_manifest") or []
                if isinstance(value, dict) and value.get("id")
            ]
            incomplete_reasons = [str(value) for value in result.get("task_incomplete_reasons") or []]
            final_artifact = await persist_task_artifact(
                task_id=task.id,
                agent_run_id=run.id,
                kind="final_report",
                content=final_answer,
                provenance={
                    "incomplete": incomplete,
                    "draft_model": result.get("task_draft_metadata") or {},
                    "quality_review": result.get("task_critic_report") or {},
                    "plan_revision": int(result.get("task_plan_revision") or request.task_plan_revision or 0),
                    "evidence_manifest": evidence_manifest,
                    "evidence_gaps": incomplete_reasons,
                },
                source_refs={"artifact_ids": [str(item["id"]) for item in evidence_manifest]},
            )
            await _complete_run_with_trace(
                repository,
                run=run,
                recorder=trace,
                status=status,
                metrics=metrics,
                result=result,
                error=error,
            )
            await tasks.complete_task(
                task.id,
                status=AgentTaskStatus.COMPLETED.value,
                reason="incomplete" if incomplete else "completed",
                final_artifact_id=final_artifact.id,
            )
        else:
            await _complete_run_with_trace(
                repository,
                run=run,
                recorder=trace,
                status=status,
                metrics=metrics,
                result=result,
                error=error,
            )
            await tasks.complete_task(task.id, status=AgentTaskStatus.FAILED.value, reason=str((error or {}).get("code") or status))
    except Exception as exc:
        logger.exception("Deep research task execution failed | task_id=%s run_id=%s", task.id, run.id)
        terminal_error = exc.to_dict() if isinstance(exc, AgentRuntimeError) else {
            "code": "deep_research_execution_failed",
            "type": type(exc).__name__,
            "raw_message": str(exc)[:1000],
            "retryable": True,
        }
        await _complete_run_with_trace(
            repository,
            run=run,
            recorder=trace,
            status=AgentRunStatus.FAILED.value,
            metrics={"duration_ms": round((time.perf_counter() - started) * 1000, 2), "error_count": 1},
            result={"agent_error": terminal_error},
            error=terminal_error,
        )
        await tasks.complete_task(
            task.id,
            status=AgentTaskStatus.FAILED.value,
            reason=str(terminal_error.get("code") or "deep_research_execution_failed"),
        )
    finally:
        heartbeat.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat
        await tasks.release_task_lease(task.id, worker_id, lease_seconds=LEASE_SECONDS)


async def run_task_worker(
    *,
    once: bool = False,
    poll_seconds: float = 1.0,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Claim and execute durable tasks until a cooperative shutdown is requested."""
    shutdown = stop_event or asyncio.Event()
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    await run_task_maintenance()
    next_maintenance = time.monotonic() + MAINTENANCE_INTERVAL_SECONDS
    while True:
        if shutdown.is_set():
            return
        task = await tasks.claim_next_task(worker_id, lease_seconds=LEASE_SECONDS)
        if task is not None:
            try:
                framework = "hermes" if task.workflow_id == "hermes_rag_agent" else "langgraph"
                wake_limit = deep_agent_budgets(framework)["wake_limit_seconds"]
                await asyncio.wait_for(execute_claimed_task(task.id, worker_id), timeout=wake_limit)
            except asyncio.TimeoutError:
                if await tasks.active_runtime_budget_exhausted(task.id):
                    await tasks.complete_task(
                        task.id,
                        status=AgentTaskStatus.FAILED.value,
                        reason="active_runtime_budget_exhausted",
                    )
                else:
                    await tasks.requeue_after_wake(task.id, reason="active_runtime_wake_limit")
            except Exception:
                logger.exception("Task runner failed before task execution could be contained | task_id=%s", task.id)
                with suppress(Exception):
                    await tasks.complete_task(
                        task.id,
                        status=AgentTaskStatus.FAILED.value,
                        reason="deep_research_runner_failed",
                    )
                with suppress(Exception):
                    await tasks.release_task_lease(task.id, worker_id)
        elif once:
            return
        else:
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=max(0.2, poll_seconds))
                return
            except asyncio.TimeoutError:
                pass
        if time.monotonic() >= next_maintenance:
            with suppress(Exception):
                await run_task_maintenance()
            next_maintenance = time.monotonic() + MAINTENANCE_INTERVAL_SECONDS
