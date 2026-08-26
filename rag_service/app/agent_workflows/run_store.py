from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import AgentRun, AgentRunEvent, AgentWorkflow, ChatTurn
from app.time_utils import parse_datetime_utc, utc_now


async def get_run(session: AsyncSession, run_id: str) -> Optional[AgentRun]:
    async with session.begin():
        return await session.get(AgentRun, run_id)


async def list_runs_for_thread(
    session: AsyncSession,
    thread_id: str,
    *,
    limit: int = 20,
    status: Optional[str] = None,
) -> list[AgentRun]:
    bounded_limit = max(1, min(int(limit), 100))
    async with session.begin():
        query = select(AgentRun).where(AgentRun.thread_id == thread_id)
        if status:
            query = query.where(AgentRun.status == status)
        result = await session.execute(
            query.order_by(AgentRun.started_at.desc(), AgentRun.id.desc()).limit(bounded_limit)
        )
        return list(result.scalars().all())


async def list_chat_turns_for_run(session: AsyncSession, run_id: str) -> list[ChatTurn]:
    async with session.begin():
        result = await session.execute(
            select(ChatTurn)
            .where(ChatTurn.agent_run_id == run_id)
            .order_by(ChatTurn.agent_run_sequence.asc(), ChatTurn.created_at.asc(), ChatTurn.id.asc())
        )
        return list(result.scalars().all())


async def create_run(
    session: AsyncSession,
    *,
    thread_id: str,
    workflow_id: str,
    workflow_version_id: Optional[str] = None,
    workflow_version: Optional[int] = None,
    framework: Optional[str],
    builder_id: Optional[str],
    definition_category: Optional[str] = None,
    resolved_spec_json: Dict[str, Any],
    user_id: Optional[str] = None,
    checkpoint_thread_id: Optional[str] = None,
    runtime_binding_json: Optional[Dict[str, Any]] = None,
    running_status: str = "running",
    run_metadata_json: Optional[Dict[str, Any]] = None,
) -> AgentRun:
    async with session.begin():
        workflow = await session.get(AgentWorkflow, workflow_id)
        if workflow is None:
            raise ValueError(f"Agent workflow is unavailable: {workflow_id}")
        persisted_framework = str(getattr(workflow, "framework", "") or "").strip()
        persisted_builder_id = str(getattr(workflow, "builder_id", "") or "").strip()
        if not persisted_framework or not persisted_builder_id:
            raise ValueError("Agent workflow runtime identity is required")
        if framework is not None and str(framework) != persisted_framework:
            raise ValueError("Run framework identity conflicts with the persisted workflow")
        if builder_id is not None and str(builder_id) != persisted_builder_id:
            raise ValueError("Run builder identity conflicts with the persisted workflow")

        run_metadata: Dict[str, Any] = dict(run_metadata_json or {})
        run_metadata.setdefault("checkpoint_boundary_available", False)
        if workflow_version_id is not None:
            run_metadata["workflow_version_id"] = workflow_version_id
        if workflow_version is not None:
            run_metadata["workflow_version"] = workflow_version
        run_id = str(uuid.uuid4())
        is_langgraph = persisted_framework == "langgraph"
        effective_checkpoint_thread_id = (checkpoint_thread_id or run_id) if is_langgraph else checkpoint_thread_id
        default_runtime_binding = (
            {
                "binding_type": "langgraph_checkpoint",
                "payload": {"checkpoint_thread_id": effective_checkpoint_thread_id},
            }
            if is_langgraph
            else {
                "binding_type": f"{persisted_framework}_session",
                "payload": {},
            }
        )
        run = AgentRun(
            id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            workflow_id=workflow_id,
            framework=persisted_framework,
            builder_id=persisted_builder_id,
            definition_category=definition_category,
            run_metadata_json=run_metadata,
            resolved_spec_json=resolved_spec_json,
            status=running_status,
            checkpoint_thread_id=effective_checkpoint_thread_id,
            runtime_binding_json=dict(runtime_binding_json or default_runtime_binding),
            runtime_binding_status="active",
            started_at=utc_now(),
        )
        session.add(run)
        await session.flush()
        await session.refresh(run)
    return run


async def delete_run(session: AsyncSession, run_id: str) -> bool:
    """Delete one exact agent run without applying retention cutoffs."""

    async with session.begin():
        run = await session.get(AgentRun, run_id)
        if run is None:
            return False
        await session.delete(run)
        await session.flush()
        return True


async def complete_run(
    session: AsyncSession,
    run_id: str,
    *,
    status: str,
    metrics_json: Optional[Dict[str, Any]] = None,
    error_json: Optional[Dict[str, Any]] = None,
    debug_trace_json: Optional[Dict[str, Any]] = None,
    completed_at: Optional[datetime] = None,
) -> Optional[AgentRun]:
    async with session.begin():
        run = await session.get(AgentRun, run_id)
        if not run:
            return None
        run.status = status
        run.completed_at = completed_at or utc_now()
        replace_jsonb_field(run, "metrics_json", metrics_json or {})
        if error_json is not None:
            replace_jsonb_field(run, "error_json", error_json)
        if debug_trace_json is not None:
            replace_jsonb_field(run, "debug_trace_json", debug_trace_json)
        await session.flush()
        await session.refresh(run)
        return run


async def set_run_debug_trace(
    session: AsyncSession,
    run_id: str,
    debug_trace_json: Dict[str, Any],
) -> Optional[AgentRun]:
    async with session.begin():
        run = await session.get(AgentRun, run_id)
        if not run:
            return None
        replace_jsonb_field(run, "debug_trace_json", debug_trace_json)
        await session.flush()
        await session.refresh(run)
        return run


async def append_run_event(
    session: AsyncSession,
    *,
    run_id: str,
    event_id: str,
    sequence: Optional[int],
    attempt: int,
    kind: str,
    payload_json: Dict[str, Any],
    occurred_at: Any = None,
    trace_id: Optional[str] = None,
    terminal: bool = False,
    source_metadata_json: Optional[Dict[str, Any]] = None,
) -> bool:
    async with session.begin():
        run = await session.execute(
            select(AgentRun.id).where(AgentRun.id == run_id).with_for_update()
        )
        if run.scalar_one_or_none() is None:
            return False
        existing = await session.execute(
            select(AgentRunEvent.id).where(
                AgentRunEvent.agent_run_id == run_id,
                AgentRunEvent.event_id == event_id,
            )
        )
        if existing.scalar_one_or_none() is not None:
            return False
        current = await session.execute(
            select(func.max(AgentRunEvent.sequence)).where(AgentRunEvent.agent_run_id == run_id)
        )
        canonical_sequence = max(0, int(current.scalar_one_or_none() or 0)) + 1
        source_metadata = dict(source_metadata_json or {})
        if sequence:
            source_metadata.setdefault("source_sequence", int(sequence))
        values = {
            "id": str(uuid.uuid4()),
            "agent_run_id": run_id,
            "event_id": event_id,
            "sequence": canonical_sequence,
            "attempt": max(1, int(attempt or 1)),
            "kind": kind,
            "payload_json": dict(payload_json or {}),
            "occurred_at": parse_datetime_utc(occurred_at) if occurred_at else None,
            "trace_id": trace_id,
            "terminal": bool(terminal),
            "source_metadata_json": source_metadata,
            "created_at": utc_now(),
        }
        statement = pg_insert(AgentRunEvent).values(**values)
        result = await session.execute(statement)
        return bool(result.rowcount)


async def list_run_events(session: AsyncSession, run_id: str) -> list[AgentRunEvent]:
    async with session.begin():
        result = await session.execute(
            select(AgentRunEvent)
            .where(AgentRunEvent.agent_run_id == run_id)
            .order_by(AgentRunEvent.attempt.asc(), AgentRunEvent.sequence.asc(), AgentRunEvent.created_at.asc())
        )
        return list(result.scalars().all())
