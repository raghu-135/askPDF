from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import AgentRun, AgentRunStatus
from app.time_utils import utc_now


CHECKPOINT_PRUNABLE_RUN_STATUSES = {
    AgentRunStatus.COMPLETED.value,
    AgentRunStatus.CLARIFICATION.value,
    AgentRunStatus.FAILED.value,
    AgentRunStatus.REJECTED.value,
    AgentRunStatus.EXPIRED.value,
    AgentRunStatus.CANCELLED.value,
}


async def prune_runs_before(
    session: AsyncSession,
    cutoff: datetime,
    *,
    statuses: list[str],
    thread_id: Optional[str] = None,
    limit: int = 1000,
) -> list[str]:
    if not statuses:
        raise ValueError("statuses must contain at least one status")
    bounded_limit = max(1, min(int(limit), 1000))
    async with session.begin():
        query = (
            select(AgentRun.id)
            .where(AgentRun.started_at < cutoff)
            .where(AgentRun.status.in_(statuses))
        )
        if thread_id is not None:
            query = query.where(AgentRun.thread_id == thread_id)
        query = query.order_by(AgentRun.started_at.asc(), AgentRun.id.asc()).limit(bounded_limit)
        result = await session.execute(query)
        run_ids = list(result.scalars().all())
        if not run_ids:
            return []
        await session.execute(delete(AgentRun).where(AgentRun.id.in_(run_ids)))
        return run_ids


async def prune_checkpoints_for_runs_before(
    session: AsyncSession,
    cutoff: datetime,
    *,
    statuses: Optional[list[str]] = None,
    thread_id: Optional[str] = None,
    limit: int = 1000,
    checkpointer: Any = None,
) -> list[str]:
    requested_statuses = statuses or sorted(CHECKPOINT_PRUNABLE_RUN_STATUSES)
    if not requested_statuses:
        raise ValueError("statuses must contain at least one status")
    invalid_statuses = sorted(set(requested_statuses) - CHECKPOINT_PRUNABLE_RUN_STATUSES)
    if invalid_statuses:
        raise ValueError(
            "checkpoint cleanup is only allowed for terminal run statuses; "
            f"invalid statuses: {', '.join(invalid_statuses)}"
        )
    bounded_limit = max(1, min(int(limit), 1000))
    async with session.begin():
        query = (
            select(AgentRun.checkpoint_thread_id)
            .where(AgentRun.started_at < cutoff)
            .where(AgentRun.status.in_(requested_statuses))
            .where(AgentRun.checkpoint_thread_id.isnot(None))
        )
        if thread_id is not None:
            query = query.where(AgentRun.thread_id == thread_id)
        result = await session.execute(
            query.order_by(AgentRun.started_at.asc(), AgentRun.id.asc()).limit(bounded_limit)
        )
        checkpoint_thread_ids = list(result.scalars().all())
    from app.runtime.langgraph.checkpointing import delete_agent_checkpoints
    return await delete_agent_checkpoints(checkpoint_thread_ids, checkpointer=checkpointer)


async def fail_stale_running_runs(
    session: AsyncSession,
    cutoff: datetime,
    *,
    thread_id: Optional[str] = None,
    limit: int = 1000,
) -> list[str]:
    bounded_limit = max(1, min(int(limit), 1000))
    async with session.begin():
        query = (
            select(AgentRun)
            .where(AgentRun.started_at < cutoff)
            .where(AgentRun.status == AgentRunStatus.RUNNING.value)
        )
        if thread_id is not None:
            query = query.where(AgentRun.thread_id == thread_id)
        query = query.order_by(AgentRun.started_at.asc(), AgentRun.id.asc()).limit(bounded_limit)
        result = await session.execute(query)
        runs = list(result.scalars().all())
        completed_at = utc_now()
        for run in runs:
            run.status = AgentRunStatus.FAILED.value
            run.completed_at = completed_at
            replace_jsonb_field(
                run,
                "error_json",
                {
                    "code": "agent_run_stale",
                    "raw_message": "Agent run was still running past the stale-run cutoff.",
                    "retryable": True,
                },
            )
            metrics = dict(run.metrics_json or {})
            metrics["error_count"] = max(int(metrics.get("error_count") or 0), 1)
            replace_jsonb_field(run, "metrics_json", metrics)
        return [run.id for run in runs]
