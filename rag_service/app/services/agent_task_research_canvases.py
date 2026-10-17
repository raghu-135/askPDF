"""Research canvas deliverables owned by durable agent tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.models_sqlmodel import AgentRun, AgentTaskArtifact
from app.models.canvas import RESEARCH_CANVAS_ARTIFACT_KIND
from app.time_utils import maybe_iso_utc_z


async def list_research_canvas_refs_for_task_ids(
    thread_id: str,
    task_ids: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    """Map each task id to its published research canvases (newest first)."""

    ids = [str(value) for value in task_ids if value]
    if not ids:
        return {}

    from app.db.repositories.canvas_repo_sqlmodel import CanvasRepository

    current_ids = {
        item.id
        for item in await CanvasRepository().list_for_thread(thread_id, current_only=True)
    }

    async with async_session_maker() as session:
        artifacts = list(
            (
                await session.execute(
                    select(AgentTaskArtifact)
                    .where(
                        AgentTaskArtifact.task_id.in_(ids),
                        AgentTaskArtifact.kind == RESEARCH_CANVAS_ARTIFACT_KIND,
                        AgentTaskArtifact.validity == "valid",
                        AgentTaskArtifact.deleted_at.is_(None),
                    )
                    .order_by(AgentTaskArtifact.created_at.desc(), AgentTaskArtifact.id.desc())
                )
            ).scalars().all()
        )
        run_ids = {artifact.agent_run_id for artifact in artifacts if artifact.agent_run_id}
        attempts: dict[str, int] = {}
        if run_ids:
            for run in (
                await session.execute(select(AgentRun).where(AgentRun.id.in_(run_ids)))
            ).scalars():
                attempts[run.id] = int(run.task_attempt or 0)

    grouped: dict[str, list[dict[str, Any]]] = {task_id: [] for task_id in ids}
    for artifact in artifacts:
        task_id = str(artifact.task_id or "")
        if task_id not in grouped:
            continue
        summary = dict(artifact.summary_json or {})
        grouped[task_id].append(
            {
                "id": artifact.id,
                "title": str(summary.get("title") or "Research canvas"),
                "run_id": artifact.agent_run_id,
                "attempt": attempts.get(str(artifact.agent_run_id or "")),
                "created_at": maybe_iso_utc_z(artifact.created_at),
                "is_current_on_thread": artifact.id in current_ids,
                "chat_turn_id": artifact.chat_turn_id,
                "supersedes_id": artifact.supersedes_id,
            }
        )
    return grouped


async def attach_research_canvases(payload: dict[str, Any], *, thread_id: str, task_id: str) -> dict[str, Any]:
    refs = await list_research_canvas_refs_for_task_ids(thread_id, [task_id])
    payload["research_canvases"] = refs.get(task_id, [])
    return payload
