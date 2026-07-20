from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import AgentRun, ChatTurn
from app.time_utils import utc_now


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
    resolved_spec_json: Dict[str, Any],
    user_id: Optional[str] = None,
    checkpoint_thread_id: Optional[str] = None,
    running_status: str = "running",
    run_metadata_json: Optional[Dict[str, Any]] = None,
) -> AgentRun:
    run_metadata: Dict[str, Any] = dict(run_metadata_json or {})
    if workflow_version_id is not None:
        run_metadata["workflow_version_id"] = workflow_version_id
    if workflow_version is not None:
        run_metadata["workflow_version"] = workflow_version
    run_id = str(uuid.uuid4())
    run = AgentRun(
        id=run_id,
        thread_id=thread_id,
        user_id=user_id,
        workflow_id=workflow_id,
        run_metadata_json=run_metadata,
        resolved_spec_json=resolved_spec_json,
        status=running_status,
        checkpoint_thread_id=checkpoint_thread_id or run_id,
        started_at=utc_now(),
    )
    async with session.begin():
        session.add(run)
        await session.flush()
        await session.refresh(run)
    return run


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
