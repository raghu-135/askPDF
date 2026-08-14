"""Compatibility translation between current workflow values and neutral contracts.

This module is intentionally the only Phase 1 contract module allowed to know
that the active implementation is LangGraph. It does not change execution or
select runtimes; Phase 2 will add that routing boundary.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from app.runtime.catalog import definition_from_workflow
from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
)
from app.runtime.errors import RuntimeError


def definition_for_workflow(workflow: Any) -> AgentDefinition:
    return definition_from_workflow(workflow)


def request_for_run(
    run: Any,
    *,
    input: Optional[Mapping[str, Any]] = None,
    options: Optional[Mapping[str, Any]] = None,
    trace_id: Optional[str] = None,
) -> AgentRuntimeRequest:
    metadata = run.run_metadata_json if isinstance(getattr(run, "run_metadata_json", None), dict) else {}
    binding = continuation_from_run(run)
    return AgentRuntimeRequest(
        run_id=str(run.id),
        thread_id=str(run.thread_id),
        definition_id=str(run.workflow_id),
        framework=str(getattr(run, "framework", None) or metadata.get("framework") or "langgraph"),
        builder_id=str(getattr(run, "builder_id", None) or metadata.get("builder_id") or "langgraph_graph"),
        input=dict(input or {}),
        options=dict(options or {}),
        task_id=getattr(run, "task_id", None),
        parent_run_id=getattr(run, "parent_run_id", None),
        continuation=binding,
        trace_id=trace_id,
    )


def continuation_from_run(run: Any) -> Optional[ContinuationBinding]:
    if getattr(run, "runtime_binding_status", "active") == "legacy_unresolved":
        return None
    payload = getattr(run, "runtime_binding_json", None)
    if isinstance(payload, dict) and payload:
        return ContinuationBinding(
            binding_type=str(payload.get("binding_type") or "langgraph"),
            payload=dict(payload.get("payload") or {}),
            binding_version=int(payload.get("binding_version") or getattr(run, "runtime_binding_version", 1) or 1),
            runtime_version=payload.get("runtime_version"),
        )
    checkpoint_id = getattr(run, "checkpoint_thread_id", None)
    if checkpoint_id:
        return ContinuationBinding(binding_type="langgraph_checkpoint", payload={"checkpoint_thread_id": str(checkpoint_id)})
    return None


def result_from_legacy(result: Mapping[str, Any]) -> AgentRuntimeResult:
    status = str(result.get("status") or ("clarification" if result.get("clarification_options") else "completed"))
    clarification = None
    if result.get("clarification_options"):
        clarification = {"options": list(result["clarification_options"])}
    interruption = result.get("pending_interrupt") if isinstance(result.get("pending_interrupt"), dict) else None
    error = result.get("agent_error") if isinstance(result.get("agent_error"), dict) else None
    return AgentRuntimeResult(
        status=status,
        output=result.get("answer") if "answer" in result else result.get("final_output"),
        clarification=clarification,
        interruption=interruption,
        usage=dict(result.get("usage") or result.get("metrics") or {}),
        runtime_metadata={
            **{
                key: result[key]
                for key in ("agent_run_id", "checkpoint_thread_id", "agent_workflow_id", "agent_workflow_version")
                if key in result
            },
            "legacy_result": dict(result),
        },
        error=error,
    )


def legacy_result_from_runtime(result: AgentRuntimeResult) -> dict[str, Any]:
    """Compatibility projection used while existing control-plane code migrates."""

    legacy = result.runtime_metadata.get("legacy_result") if isinstance(result.runtime_metadata, Mapping) else None
    if isinstance(legacy, dict):
        return dict(legacy)
    return {
        "status": result.status,
        "answer": result.output,
        "clarification_options": list((result.clarification or {}).get("options") or []),
        "pending_interrupt": dict(result.interruption or {}),
        "agent_error": dict(result.error or {}),
    }


def event_from_legacy(
    event: Mapping[str, Any],
    *,
    run_id: str,
    sequence: int,
    event_id: Optional[str] = None,
) -> AgentRuntimeEvent:
    data = dict(event.get("data") or {})
    kind = str(event.get("event") or event.get("kind") or "runtime.event")
    return AgentRuntimeEvent(
        event_id=str(event_id or data.get("event_id") or f"{run_id}:{sequence}"),
        run_id=run_id,
        sequence=sequence,
        kind=kind,
        payload=data,
        occurred_at=data.get("occurred_at") or data.get("timestamp"),
        terminal=kind in {"run.completed", "run.failed", "run.cancelled", "run.terminal"},
        trace_id=data.get("trace_id"),
    )


def error_from_exception(exc: BaseException, *, code: str = "agent_runtime_failed", retryable: bool = False) -> RuntimeError:
    return RuntimeError.from_exception(exc, code=code, retryable=retryable)
