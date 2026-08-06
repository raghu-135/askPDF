from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any, AsyncIterator, Dict, Optional

from langgraph.types import Command
from sqlalchemy import delete
from sqlalchemy.future import select

from app.agent_workflows.compiler import WorkflowCompiler
from app.agent_workflows.debug_trace import AgentTraceRecorder, merge_debug_payloads
from app.agent_workflows.metrics import build_run_metrics
from app.agent_workflows.parallel_runtime import normalized_parallel_policy
from app.agent_workflows.workflow_runtime import workflow_runtime_features
from app.agent_workflows.enums import WorkflowNodeType
from app.agent_workflows.router_runtime import (
    _pending_interrupt_from_result,
    _runtime_config,
    _without_runtime_keys,
)
from app.agent_workflows.trace_sanitization import _bounded_value
from app.agent_workflows.trace_details import final_output_from_result
from app.agent_workflows.validator import WorkflowValidator
from app.agent_workflows.repository import AgentWorkflowRepository
from app.db import AgentRunStatus
from app.db.connection_sqlmodel import async_session_maker
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import AgentRun
from app.models.llm_server_client import DEFAULT_TOKEN_BUDGET
from app.time_utils import utc_now


RUN_KIND = "builder_test"
TERMINAL = {
    AgentRunStatus.COMPLETED.value,
    AgentRunStatus.CLARIFICATION.value,
    AgentRunStatus.FAILED.value,
    AgentRunStatus.REJECTED.value,
    AgentRunStatus.EXPIRED.value,
    AgentRunStatus.CANCELLED.value,
}


def spec_fingerprint(spec: Dict[str, Any]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:20]


async def latest_builder_test(builder_session_id: str, base_workflow_id: Optional[str] = None) -> Optional[AgentRun]:
    async with async_session_maker() as session:
        async with session.begin():
            query = select(AgentRun).where(
                AgentRun.run_metadata_json["run_kind"].astext == RUN_KIND,
                AgentRun.run_metadata_json["builder_session_id"].astext == builder_session_id,
            )
            if base_workflow_id:
                query = query.where(AgentRun.run_metadata_json["base_workflow_id"].astext == base_workflow_id)
            result = await session.execute(query.order_by(AgentRun.started_at.desc(), AgentRun.id.desc()).limit(1))
            return result.scalars().first()


async def delete_previous_builder_tests(builder_session_id: str, *, keep_run_id: Optional[str] = None) -> list[str]:
    async with async_session_maker() as session:
        async with session.begin():
            query = select(AgentRun).where(
                AgentRun.run_metadata_json["run_kind"].astext == RUN_KIND,
                AgentRun.run_metadata_json["builder_session_id"].astext == builder_session_id,
            )
            if keep_run_id:
                query = query.where(AgentRun.id != keep_run_id)
            result = await session.execute(query)
            runs = list(result.scalars().all())
            terminal_runs = [run for run in runs if run.status in TERMINAL]
            active_runs = [run for run in runs if run.status not in TERMINAL]
            for run in active_runs:
                metadata = dict(run.run_metadata_json or {})
                metadata["cancel_requested"] = True
                metadata["superseded"] = True
                replace_jsonb_field(run, "run_metadata_json", metadata)
            if terminal_runs:
                await session.execute(delete(AgentRun).where(AgentRun.id.in_([run.id for run in terminal_runs])))
            return [str(run.checkpoint_thread_id or run.id) for run in terminal_runs]


async def request_builder_test_cancel(run_id: str) -> Optional[AgentRun]:
    async with async_session_maker() as session:
        async with session.begin():
            run = await session.get(AgentRun, run_id)
            if run is None or (run.run_metadata_json or {}).get("run_kind") != RUN_KIND:
                return None
            metadata = dict(run.run_metadata_json or {})
            metadata["cancel_requested"] = True
            replace_jsonb_field(run, "run_metadata_json", metadata)
            await session.flush()
            await session.refresh(run)
            return run


async def cancel_requested(run_id: str) -> bool:
    async with async_session_maker() as session:
        async with session.begin():
            run = await session.get(AgentRun, run_id)
            return bool(run and (run.run_metadata_json or {}).get("cancel_requested"))


async def persist_terminal_run(
    run_id: str,
    *,
    status: str,
    result: Dict[str, Any],
    duration_ms: float,
    recorder: AgentTraceRecorder,
    error: Optional[Dict[str, Any]] = None,
) -> AgentRun:
    metrics = build_run_metrics(result, duration_ms=duration_ms)
    async with async_session_maker() as session:
        async with session.begin():
            run = await session.get(AgentRun, run_id)
            if run is None:
                raise RuntimeError("Builder test run disappeared")
            run.status = status
            run.completed_at = utc_now()
            replace_jsonb_field(run, "metrics_json", metrics)
            if error is not None:
                replace_jsonb_field(run, "error_json", error)
            debug = recorder.finalize(
                run=run,
                chat_turn_id=None,
                metrics=metrics,
                route=result.get("route"),
                route_reason=result.get("route_reason"),
                error=error,
                result=result,
            )
            if isinstance(run.debug_trace_json, dict):
                debug = merge_debug_payloads(
                    run.debug_trace_json,
                    debug,
                    resolved_spec=run.resolved_spec_json if isinstance(run.resolved_spec_json, dict) else {},
                    run_status=run.status,
                    completed_at=run.completed_at,
                    metrics=metrics,
                )
            replace_jsonb_field(run, "debug_trace_json", debug)
            await session.flush()
            await session.refresh(run)
            return run


def initial_studio_state(
    *,
    run_id: str,
    thread_id: str,
    spec: Dict[str, Any],
    request: Any,
    embedding_model: str,
) -> Dict[str, Any]:
    config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
    features = workflow_runtime_features(spec)
    parallel_enabled = bool(features.get("supports_parallel_dispatch"))
    parallel_policy = normalized_parallel_policy(config.get("parallel_policy"))
    graph_nodes = ((config.get("graph") or {}).get("nodes") or []) if isinstance(config.get("graph"), dict) else []
    transient_messages = getattr(request, "transient_messages", None) or []
    transient_history = "\n".join(
        f"{str(getattr(message, 'role', '')).capitalize()}: {str(getattr(message, 'content', '')).strip()}"
        for message in transient_messages
        if str(getattr(message, "content", "")).strip()
    )
    return {
        "agent_run_id": run_id,
        "workflow_id": spec.get("workflow_id"),
        "thread_id": thread_id,
        "question": request.question,
        "llm_model": request.llm_model,
        "embedding_model": embedding_model,
        "context_window": request.context_window or DEFAULT_TOKEN_BUDGET,
        "use_web_search": bool(request.use_web_search),
        "use_reranker": request.use_reranker if request.use_reranker is not None else True,
        "system_role": request.system_role_override or "",
        "tool_instructions": request.tool_instructions_override or {},
        "custom_instructions": request.custom_instructions_override or "",
        "allowed_tool_ids": list(config.get("allowed_tool_ids") or []),
        "hitl_policy": dict(config.get("hitl_policy") or {}),
        "loop_policy": dict(config.get("loop_policy") or {}),
        "context_policy": dict(config.get("context_policy") or {}),
        "prefetch_policy": dict(config.get("prefetch_policy") or {}),
        "parallel_enabled": parallel_enabled,
        "parallel_policy": parallel_policy,
        "parallel_aggregator_id": next((str(node.get("id")) for node in graph_nodes if isinstance(node, dict) and node.get("type") == WorkflowNodeType.AGGREGATOR.value), ""),
        "dispatch_aggregator_id": next((str(node.get("id")) for node in graph_nodes if isinstance(node, dict) and node.get("type") == WorkflowNodeType.AGGREGATOR.value), ""),
        "worker_result_packets": [],
        "parallel_runtime_override": True,
        "parallel_evidence_deltas": [],
        "parallel_document_source_deltas": [],
        "parallel_web_source_deltas": [],
        "parallel_chat_id_deltas": [],
        "parallel_memory_ref_deltas": [],
        "parallel_timeline_ref_deltas": [],
        "parallel_node_event_deltas": [],
        "parallel_tool_event_deltas": [],
        "parallel_error_deltas": [],
        "parallel_skipped_work_deltas": [],
        "parallel_visit_records": [],
        "parallel_attempt_records": [],
        "node_visit_counts": {},
        "node_visit_sequence": [],
        "evidence_packets": [],
        "hitl_interrupt_counts": {},
        "replans": max(1, int(config.get("replans", request.replans or 1))),
        "replan_count": 0,
        "replan_history": [],
        "client_timezone": request.client_timezone,
        "client_locale": request.client_locale,
        "client_now_iso": request.client_now_iso,
        "transient_history_text": transient_history,
        "document_sources": [],
        "web_sources": [],
        "used_chat_ids": [],
        "node_events": [],
        "tool_events": [],
        "errors": [],
    }


async def stream_builder_test(
    *,
    run: AgentRun,
    request: Any,
    embedding_model: str,
    checkpointer: Any,
    resume_decision: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Dict[str, Any]]:
    spec = run.resolved_spec_json if isinstance(run.resolved_spec_json, dict) else {}
    WorkflowValidator().validate(spec)
    recorder = AgentTraceRecorder(run)
    queue: asyncio.Queue = asyncio.Queue()
    telemetry: Dict[str, Any] = {"node_events": [], "tool_events": []}
    config = _runtime_config(
        app_thread_id=run.thread_id,
        checkpoint_thread_id=str(run.checkpoint_thread_id or run.id),
        embedding_model=embedding_model,
        context_window=request.context_window or DEFAULT_TOKEN_BUDGET,
        use_web_search=request.use_web_search,
        use_reranker=request.use_reranker,
        telemetry_sink=telemetry,
        trace_recorder=recorder,
        max_concurrency=normalized_parallel_policy((spec.get("config") or {}).get("parallel_policy"))["max_concurrency"] if workflow_runtime_features(spec).get("supports_parallel_dispatch") else None,
    )
    config["configurable"]["studio_event_queue"] = queue
    app = WorkflowCompiler().compile(spec, checkpointer=checkpointer)
    graph_input: Any = Command(resume=resume_decision) if resume_decision is not None else initial_studio_state(
        run_id=run.id,
        thread_id=run.thread_id,
        spec=spec,
        request=request,
        embedding_model=embedding_model,
    )
    latest: Dict[str, Any] = {}
    started = time.perf_counter()

    async def consume() -> None:
        nonlocal latest
        try:
            async for chunk in app.astream(graph_input, config=config, stream_mode="values"):
                if isinstance(chunk, dict):
                    latest = chunk
            await queue.put({"event": "__done__", "data": {}})
        except asyncio.CancelledError:
            await queue.put({"event": "__cancelled__", "data": {}})
        except Exception as exc:
            await queue.put({"event": "__failed__", "data": {"error": str(exc)}})

    task = asyncio.create_task(consume())
    yield {"event": "run.started", "data": {"run_id": run.id, "spec_fingerprint": (run.run_metadata_json or {}).get("spec_fingerprint")}}
    cancel_pending = False
    finalized = False
    seen_tool_events = 0
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=12)
            except asyncio.TimeoutError:
                if await cancel_requested(run.id):
                    cancel_pending = True
                yield {"event": "heartbeat", "data": {"run_id": run.id, "cancel_pending": cancel_pending}}
                continue
            event = item["event"]
            if event == "__done__":
                current_tools = telemetry.get("tool_events") if isinstance(telemetry.get("tool_events"), list) else []
                full_tools = recorder.tool_details()
                for index, tool_event in enumerate(current_tools[seen_tool_events:], start=seen_tool_events):
                    yield {"event": "tool.completed", "data": {**_bounded_value(tool_event), "detail": full_tools[index] if index < len(full_tools) else None}}
                seen_tool_events = len(current_tools)
                break
            if event == "__failed__":
                raise RuntimeError(str(item.get("data", {}).get("error") or "Builder test failed"))
            if event == "__cancelled__":
                cancel_pending = True
                break
            event_data = item.get("data") or {}
            detail = event_data.get("detail") if isinstance(event_data, dict) else None
            bounded_event_data = _bounded_value({key: value for key, value in event_data.items() if key != "detail"})
            if detail is not None:
                bounded_event_data["detail"] = detail
            yield {"event": event, "data": bounded_event_data}
            current_tools = telemetry.get("tool_events") if isinstance(telemetry.get("tool_events"), list) else []
            full_tools = recorder.tool_details()
            for index, tool_event in enumerate(current_tools[seen_tool_events:], start=seen_tool_events):
                yield {"event": "tool.completed", "data": {**_bounded_value(tool_event), "detail": full_tools[index] if index < len(full_tools) else None}}
            seen_tool_events = len(current_tools)
            if await cancel_requested(run.id):
                cancel_pending = True
            if cancel_pending and event in {"node.completed", "node.skipped", "node.failed"}:
                task.cancel()
                break

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        if cancel_pending:
            partial = _without_runtime_keys(latest)
            await persist_terminal_run(
                run.id,
                status=AgentRunStatus.CANCELLED.value,
                result=partial,
                duration_ms=duration_ms,
                recorder=recorder,
            )
            finalized = True
            yield {"event": "run.canceled", "data": {"run_id": run.id, "duration_ms": duration_ms}}
            return

        pending = _pending_interrupt_from_result(latest, checkpoint_thread_id=str(run.checkpoint_thread_id or run.id))
        result = _without_runtime_keys(latest)
        if pending:
            recorder.record_interrupted_snapshot(interrupt=pending, state=result)
            metrics = build_run_metrics(result, duration_ms=duration_ms)
            debug = recorder.finalize(run=run, chat_turn_id=None, metrics=metrics, route=result.get("route"), route_reason=result.get("route_reason"), result=result)
            if isinstance(run.debug_trace_json, dict):
                debug = merge_debug_payloads(
                    run.debug_trace_json,
                    debug,
                    resolved_spec=run.resolved_spec_json if isinstance(run.resolved_spec_json, dict) else {},
                    run_status=run.status,
                    completed_at=run.completed_at,
                    metrics=metrics,
                )
            stored = await AgentWorkflowRepository().mark_run_awaiting_human(
                run.id,
                pending,
                metrics_json=metrics,
                debug_trace_json=debug,
            )
            finalized = True
            pending = dict(stored.pending_interrupt_json or pending) if stored is not None else pending
            yield {"event": "interrupt.created", "data": _bounded_value(pending)}
            return

        status = AgentRunStatus.CLARIFICATION.value if result.get("clarification_options") else AgentRunStatus.COMPLETED.value
        await persist_terminal_run(run.id, status=status, result=result, duration_ms=duration_ms, recorder=recorder)
        finalized = True
        yield {
            "event": "run.completed",
            "data": {
                "run_id": run.id,
                "duration_ms": duration_ms,
                "status": status,
                "answer": final_output_from_result(result).get("answer"),
                "final_output": final_output_from_result(result),
                "route": result.get("route"),
                "route_reason": result.get("route_reason"),
            },
        }
    except Exception as exc:
        task.cancel()
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        error = {"code": "builder_test_failed", "raw_message": str(exc), "retryable": True}
        partial = _without_runtime_keys(latest)
        partial["agent_error"] = error
        await persist_terminal_run(
            run.id,
            status=AgentRunStatus.FAILED.value,
            result=partial,
            duration_ms=duration_ms,
            recorder=recorder,
            error=error,
        )
        finalized = True
        yield {"event": "run.failed", "data": _bounded_value({"run_id": run.id, "error": error, "duration_ms": duration_ms})}
    finally:
        if not task.done():
            task.cancel()
        if not finalized:
            try:
                await persist_terminal_run(
                    run.id,
                    status=AgentRunStatus.CANCELLED.value,
                    result=_without_runtime_keys(latest),
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    recorder=recorder,
                )
            except Exception:
                pass
