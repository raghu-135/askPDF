"""Explicit publication of a deep-research final report into thread chat."""

from __future__ import annotations

from typing import Any

from sqlalchemy.future import select

from app.db import ChatTurnStatus, create_chat_turn, increment_qa_stats, update_message_context_compact
from app.db.connection_sqlmodel import async_session_maker
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import AgentTask, AgentTaskArtifact, ChatTurn
from app.rag.indexer import index_chat_memory_for_thread
from app.services import agent_task_repository as repository
from app.services.content_store import get_content_store
from app.services.embedding_model_service import (
    EmbeddingModelResolutionError,
    EmbeddingModelUnavailableError,
    require_thread_embedding_ready,
)


async def publish_final_report_to_chat(*, task_id: str, artifact_id: str, thread_id: str) -> dict[str, Any]:
    """Create one chat turn from a valid final report, or return the existing published turn."""

    task = await repository.get_task(task_id, thread_id=thread_id)
    if task is None:
        raise repository.AgentTaskConflict("task_not_found", "Agent task not found")
    artifact = await repository.get_artifact(task_id, artifact_id)
    if (
        artifact is None
        or artifact.kind != "final_report"
        or artifact.validity != "valid"
        or artifact.deleted_at is not None
    ):
        raise repository.AgentTaskConflict("final_report_not_found", "A valid final report was not found")

    existing = await _published_turn(dict(artifact.provenance_json or {}))
    if existing is not None:
        return _publication_payload(existing, duplicate=True)

    try:
        content = (await get_content_store().read(artifact.object_key)).decode("utf-8", errors="replace").strip()
    except FileNotFoundError as exc:
        raise repository.AgentTaskConflict("final_report_missing_content", "The final report content is unavailable") from exc
    if not content:
        raise repository.AgentTaskConflict("final_report_empty", "The final report has no content to add to chat")

    config = dict(task.config_json or {})
    turn = await create_chat_turn(
        thread_id=thread_id,
        question=str(task.objective or "").strip() or "Deep Research result",
        answer=content,
        status=ChatTurnStatus.COMPLETED.value,
        metadata={
            "published_from_task": True,
            "agent_task_id": task.id,
            "agent_workflow_id": task.workflow_id,
            "final_report_artifact_id": artifact.id,
        },
        agent_run_id=artifact.agent_run_id,
        agent_run_turn_kind="task_final_published",
        agent_run_sequence=0,
    )
    persisted = await _remember_publication(
        task_id=task.id,
        artifact_id=artifact.id,
        chat_turn_id=turn.id,
    )
    if persisted.id != turn.id:
        from app.db import delete_message_pair

        await delete_message_pair(f"{turn.id}:assistant")
        return _publication_payload(persisted, duplicate=True)

    await _index_published_turn(task=task, turn=persisted, question=str(task.objective or ""), answer=content, config=config)
    return _publication_payload(persisted, duplicate=False)


async def _published_turn(provenance: dict[str, Any]) -> ChatTurn | None:
    turn_id = str(provenance.get("published_chat_turn_id") or "").strip()
    if not turn_id:
        return None
    from app.db import get_message_repo

    return await get_message_repo().get_turn(turn_id)


def _publication_payload(turn: ChatTurn, *, duplicate: bool) -> dict[str, Any]:
    return {
        "chat_turn_id": turn.id,
        "user_message_id": f"{turn.id}:user",
        "assistant_message_id": f"{turn.id}:assistant",
        "duplicate": duplicate,
    }


async def _remember_publication(*, task_id: str, artifact_id: str, chat_turn_id: str) -> ChatTurn:
    from app.db import get_message_repo

    async with async_session_maker() as session:
        async with session.begin():
            task = (
                await session.execute(select(AgentTask).where(AgentTask.id == task_id).with_for_update())
            ).scalar_one_or_none()
            artifact = (
                await session.execute(
                    select(AgentTaskArtifact)
                    .where(
                        AgentTaskArtifact.id == artifact_id,
                        AgentTaskArtifact.task_id == task_id,
                    )
                    .with_for_update()
                )
            ).scalar_one_or_none()
            if task is None or artifact is None or artifact.kind != "final_report" or artifact.validity != "valid":
                raise repository.AgentTaskConflict("final_report_not_found", "A valid final report was not found")
            provenance = dict(artifact.provenance_json or {})
            existing_id = str(provenance.get("published_chat_turn_id") or "").strip()
            if existing_id:
                winner_id = existing_id
            else:
                provenance["published_chat_turn_id"] = chat_turn_id
                replace_jsonb_field(artifact, "provenance_json", provenance)
                winner_id = chat_turn_id
    turn = await get_message_repo().get_turn(winner_id)
    if turn is None:
        raise repository.AgentTaskConflict("chat_turn_missing", "The published chat turn could not be loaded")
    return turn


async def _index_published_turn(
    *,
    task: AgentTask,
    turn: ChatTurn,
    question: str,
    answer: str,
    config: dict[str, Any],
) -> None:
    llm_model = str(config.get("llm_model") or "").strip()
    try:
        embedding_context = await require_thread_embedding_ready(task.thread_id)
    except (EmbeddingModelResolutionError, EmbeddingModelUnavailableError):
        return
    if not llm_model or not embedding_context.embedding_model:
        return
    try:
        indexed = await index_chat_memory_for_thread(
            thread_id=task.thread_id,
            message_id=turn.id,
            question=question,
            answer=answer,
            embedding_model=embedding_context.embedding_model,
            llm_name=llm_model,
            context_window=config.get("context_window"),
            message_created_at=turn.completed_at or turn.created_at,
        )
        compact = indexed.get("memory_compact_text") if isinstance(indexed, dict) else None
        if compact:
            await update_message_context_compact(turn.id, compact)
        await increment_qa_stats(task.thread_id, len(question) + len(answer))
    except Exception:
        return
