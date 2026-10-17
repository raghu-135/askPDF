"""Publish deep-research canvas results into thread chat."""

from __future__ import annotations

from typing import Any

from sqlalchemy.future import select

from app.db import ChatTurnStatus, create_chat_turn
from app.db.connection_sqlmodel import async_session_maker
from app.db.models_sqlmodel import AgentTaskArtifact
from app.db.repositories.message_repo_sqlmodel import MessageRepository
from app.models.canvas import RESEARCH_CANVAS_ARTIFACT_KIND
from app.services import agent_task_repository as repository
from app.services.agent_task_chat_publish import _index_published_turn


async def ensure_deep_task_canvas_chat_turn(
    *,
    task_id: str,
    agent_run_id: str,
    thread_id: str,
    canvas_id: str,
    title: str,
    summary: str | None,
) -> dict[str, Any] | None:
    """Create a short chat turn pointing at the research canvas, or return the existing one."""

    existing = await MessageRepository().get_turn_by_agent_run_id(thread_id, agent_run_id)
    if existing is not None:
        return {"chat_turn_id": existing.id, "duplicate": True}

    task = await repository.get_task(task_id, thread_id=thread_id)
    if task is None:
        return None

    answer_parts = [str(title or "Research canvas").strip()]
    if summary and str(summary).strip():
        answer_parts.append(str(summary).strip())
    answer = "\n\n".join(part for part in answer_parts if part)
    if not answer:
        answer = str(title or "Research canvas")

    turn = await create_chat_turn(
        thread_id=thread_id,
        question=str(task.objective or "").strip() or "Deep Research result",
        answer=answer,
        status=ChatTurnStatus.COMPLETED.value,
        metadata={
            "published_from_task": True,
            "agent_task_id": task.id,
            "agent_workflow_id": task.workflow_id,
            "research_canvas_id": canvas_id,
        },
        agent_run_id=agent_run_id,
        agent_run_turn_kind="task_canvas_published",
        agent_run_sequence=0,
    )
    config = dict(task.config_json or {})
    await _index_published_turn(
        task=task,
        turn=turn,
        question=str(task.objective or ""),
        answer=answer,
        config=config,
    )
    async with async_session_maker() as session:
        async with session.begin():
            artifact = (
                await session.execute(
                    select(AgentTaskArtifact).where(
                        AgentTaskArtifact.id == canvas_id,
                        AgentTaskArtifact.kind == RESEARCH_CANVAS_ARTIFACT_KIND,
                    )
                )
            ).scalar_one_or_none()
            if artifact is not None and artifact.chat_turn_id is None:
                artifact.chat_turn_id = turn.id
                artifact.task_id = task_id
                session.add(artifact)
    return {"chat_turn_id": turn.id, "duplicate": False}


async def resolve_task_canvas_supersedes_id(thread_id: str, task_id: str) -> str | None:
    async with async_session_maker() as session:
        async with session.begin():
            rows = (
                await session.execute(
                    select(AgentTaskArtifact)
                    .where(
                        AgentTaskArtifact.thread_id == thread_id,
                        AgentTaskArtifact.task_id == task_id,
                        AgentTaskArtifact.kind == RESEARCH_CANVAS_ARTIFACT_KIND,
                        AgentTaskArtifact.validity == "valid",
                        AgentTaskArtifact.deleted_at.is_(None),
                    )
                    .order_by(AgentTaskArtifact.created_at.desc())
                )
            ).scalars().all()
    current = await _current_canvas_ids(thread_id)
    for row in rows:
        if row.id in current:
            return row.id
    return None


async def _current_canvas_ids(thread_id: str) -> set[str]:
    from app.db.repositories.canvas_repo_sqlmodel import CanvasRepository

    return {item.id for item in await CanvasRepository().list_for_thread(thread_id, current_only=True)}
