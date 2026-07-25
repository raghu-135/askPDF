"""Thread Management Service - Handles thread-level operations."""

import copy
import logging
import uuid
from typing import Any, Dict, List

from sqlalchemy.future import select

from app.db import (
    ChatTurnStatus,
    ProcessStatus,
    ThreadCloneMode,
    get_scoped_indexing_status,
    get_thread_shape,
    remove_document_from_stats,
    upsert_document_in_stats,
)
from app.db.connection_sqlmodel import async_session_maker
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import ChatTurn, Memory, MemoryEvent, Project, Thread, ThreadFile
from app.db.repositories.message_repo_sqlmodel import turn_id_from_message_id
from app.db.vector import get_vector_db
from app.time_utils import iso_utc_z, utc_now

logger = logging.getLogger(__name__)


class ThreadForkError(Exception):
    """Base exception for thread fork failures."""


class SourceThreadNotFoundError(ThreadForkError):
    """Raised when the source thread does not exist."""


class ForkMessageNotFoundError(ThreadForkError):
    """Raised when the requested fork message is not in the source thread."""


class TargetProjectEmbeddingModelMismatchError(ThreadForkError):
    """Raised when a cross-project fork would change embedding spaces."""


async def fork_thread(
    source_thread_id: str,
    message_id: str | None = None,
    name: str | None = None,
    target_project_id: str | None = None,
    memory_copy_mode: str | None = None,
) -> Dict[str, Any]:
    """Create an independent fork of a thread with soft lineage metadata."""
    forked_at = utc_now()
    forked_at_iso = iso_utc_z(forked_at)
    new_thread_id = str(uuid.uuid4())
    copied_memory_index_jobs: list[Memory] = []
    embedding_model_for_memory_index: str | None = None

    async with async_session_maker() as session:
        async with session.begin():
            source_result = await session.execute(
                select(Thread).where(Thread.id == source_thread_id)
            )
            source_thread = source_result.scalar_one_or_none()
            if not source_thread:
                raise SourceThreadNotFoundError("Source thread not found")
            embedding_model_for_memory_index = source_thread.embedding_model
            if target_project_id:
                target_project = await session.get(Project, target_project_id)
                if target_project is None:
                    raise SourceThreadNotFoundError("Target project not found")
                if (
                    target_project.id != source_thread.project_id
                    and target_project.embedding_model != source_thread.embedding_model
                ):
                    raise TargetProjectEmbeddingModelMismatchError(
                        "Thread cannot fork to a project with a different embedding model"
                    )

            turns_result = await session.execute(
                select(ChatTurn)
                .where(ChatTurn.thread_id == source_thread_id, ChatTurn.status != ChatTurnStatus.CANCELLED.value)
                .order_by(ChatTurn.created_at.asc(), ChatTurn.id.asc())
            )
            source_turns = list(turns_result.scalars().all())

            source_turn = None
            turns_to_copy = source_turns
            mode = ThreadCloneMode.FULL_THREAD.value
            if message_id:
                mode = ThreadCloneMode.FROM_MESSAGE.value
                target_turn_id = turn_id_from_message_id(message_id)
                for index, turn in enumerate(source_turns):
                    if turn.id == target_turn_id:
                        source_turn = turn
                        turns_to_copy = source_turns[: index + 1]
                        break
                if not source_turn:
                    raise ForkMessageNotFoundError("Fork message not found in source thread")

            fork_metadata = {
                "fork": {
                    "parent_thread_id": source_thread.id,
                    "parent_thread_name": source_thread.name,
                    "forked_at": forked_at_iso,
                    "source_message_id": message_id if source_turn else None,
                    "source_message_created_at": iso_utc_z(source_turn.created_at) if source_turn else None,
                    "mode": mode,
                    "memory_copy_mode": memory_copy_mode,
                }
            }
            target_project = target_project_id or source_thread.project_id
            resolved_memory_copy_mode = memory_copy_mode
            if not resolved_memory_copy_mode:
                resolved_memory_copy_mode = (
                    "project_snapshot"
                    if target_project and target_project != source_thread.project_id
                    else "thread_snapshot"
                )
            fork_metadata["fork"]["memory_copy_mode"] = resolved_memory_copy_mode
            source_metadata = copy.deepcopy(source_thread.thread_metadata or {})
            source_metadata.pop("fork_children", None)
            source_metadata.update(fork_metadata)

            forked_thread = Thread(
                id=new_thread_id,
                project_id=target_project,
                name=(name or "").strip() or f"{source_thread.name} (Fork)",
                embedding_model=source_thread.embedding_model,
                settings=copy.deepcopy(source_thread.settings or {}),
                thread_metadata=source_metadata,
                created_at=forked_at,
            )
            session.add(forked_thread)

            parent_metadata = copy.deepcopy(source_thread.thread_metadata or {})
            fork_children = parent_metadata.get("fork_children")
            if not isinstance(fork_children, list):
                fork_children = []
            if new_thread_id not in fork_children:
                fork_children.append(new_thread_id)
            parent_metadata["fork_children"] = fork_children
            replace_jsonb_field(source_thread, "thread_metadata", parent_metadata)

            for turn in turns_to_copy:
                session.add(
                    ChatTurn(
                        id=str(uuid.uuid4()),
                        thread_id=new_thread_id,
                        status=turn.status,
                        payload=copy.deepcopy(turn.payload or {}),
                        created_at=turn.created_at,
                        completed_at=turn.completed_at,
                    )
                )

            files_result = await session.execute(
                select(ThreadFile).where(ThreadFile.thread_id == source_thread_id)
            )
            source_file_associations = list(files_result.scalars().all())
            for association in source_file_associations:
                session.add(
                    ThreadFile(
                        thread_id=new_thread_id,
                        file_hash=association.file_hash,
                        added_at=association.added_at,
                        annotations=association.annotations,
                        annotations_updated_at=association.annotations_updated_at,
                    )
                )

            memory_cutoff = source_turn.created_at if source_turn else forked_at
            copied_memory_ids: list[str] = []

            if resolved_memory_copy_mode in {"thread_snapshot", "all"}:
                thread_memories_result = await session.execute(
                    select(Memory).where(
                        Memory.scope_type == "thread",
                        Memory.scope_id == source_thread_id,
                        Memory.status == "active",
                        Memory.created_at <= memory_cutoff,
                    )
                )
                for memory in thread_memories_result.scalars().all():
                    copied_id = str(uuid.uuid4())
                    copied_memory = Memory(
                        id=copied_id,
                        scope_type="thread",
                        scope_id=new_thread_id,
                        memory_type=memory.memory_type,
                        content=memory.content,
                        summary=memory.summary,
                        embedding_model=source_thread.embedding_model,
                        content_hash=memory.content_hash,
                        index_status="pending",
                        source_refs_json=copy.deepcopy(memory.source_refs_json or {}),
                        confidence=memory.confidence,
                        status=memory.status,
                        visibility=memory.visibility,
                        created_by=memory.created_by,
                        created_at=forked_at,
                        updated_at=forked_at,
                        expires_at=memory.expires_at,
                        fork_origin_json={
                            "source_memory_id": memory.id,
                            "source_thread_id": source_thread_id,
                            "forked_thread_id": new_thread_id,
                            "forked_at": forked_at_iso,
                            "copy_mode": resolved_memory_copy_mode,
                        },
                    )
                    session.add(copied_memory)
                    copied_memory_index_jobs.append(copied_memory)
                    session.add(
                        MemoryEvent(
                            memory_id=copied_id,
                            event_type="fork_snapshot",
                            actor_id=None,
                            payload_json=copy.deepcopy(copied_memory.fork_origin_json or {}),
                            created_at=forked_at,
                        )
                    )
                    copied_memory_ids.append(copied_id)

            if (
                resolved_memory_copy_mode in {"project_snapshot", "all"}
                and target_project
                and source_thread.project_id
                and target_project != source_thread.project_id
            ):
                project_memories_result = await session.execute(
                    select(Memory).where(
                        Memory.scope_type == "project",
                        Memory.scope_id == source_thread.project_id,
                        Memory.status == "active",
                    )
                )
                for memory in project_memories_result.scalars().all():
                    copied_id = str(uuid.uuid4())
                    copied_memory = Memory(
                        id=copied_id,
                        scope_type="project",
                        scope_id=target_project,
                        memory_type=memory.memory_type,
                        content=memory.content,
                        summary=memory.summary,
                        embedding_model=source_thread.embedding_model,
                        content_hash=memory.content_hash,
                        index_status="pending",
                        source_refs_json=copy.deepcopy(memory.source_refs_json or {}),
                        confidence=memory.confidence,
                        status=memory.status,
                        visibility=memory.visibility,
                        created_by=memory.created_by,
                        created_at=forked_at,
                        updated_at=forked_at,
                        expires_at=memory.expires_at,
                        fork_origin_json={
                            "source_memory_id": memory.id,
                            "source_project_id": source_thread.project_id,
                            "target_project_id": target_project,
                            "forked_thread_id": new_thread_id,
                            "forked_at": forked_at_iso,
                            "copy_mode": resolved_memory_copy_mode,
                        },
                    )
                    session.add(copied_memory)
                    copied_memory_index_jobs.append(copied_memory)
                    session.add(
                        MemoryEvent(
                            memory_id=copied_id,
                            event_type="fork_snapshot",
                            actor_id=None,
                            payload_json=copy.deepcopy(copied_memory.fork_origin_json or {}),
                            created_at=forked_at,
                        )
                    )
                    copied_memory_ids.append(copied_id)

            if copied_memory_ids:
                child_metadata = copy.deepcopy(forked_thread.thread_metadata or {})
                child_metadata.setdefault("fork", {})["copied_memory_ids"] = copied_memory_ids
                replace_jsonb_field(forked_thread, "thread_metadata", child_metadata)

            await session.flush()
            await session.refresh(forked_thread)

    # Import here to avoid creating tighter module import cycles during app startup.
    from app.db import get_thread_files, recompute_qa_stats

    try:
        await recompute_qa_stats(new_thread_id)
    except Exception as stats_err:
        logger.warning("thread stats recompute skipped after fork: %s", stats_err)
    try:
        files = await get_thread_files(new_thread_id)
    except Exception as files_err:
        logger.warning("forked thread file reload skipped: %s", files_err)
        files = []
    if copied_memory_index_jobs and embedding_model_for_memory_index:
        from app.services.memory_service import index_memory_record

        for memory in copied_memory_index_jobs:
            try:
                await index_memory_record(memory)
            except Exception as memory_index_err:
                logger.warning("forked memory indexing skipped for %s: %s", memory.id, memory_index_err)
    return {"thread": forked_thread, "files": files}


async def repair_thread_documents_meta(thread_id: str, embedding_model: str, files: List[Any]) -> None:
    """Rebuild thread document metadata for already-indexed files attached to a thread."""
    vector_db = get_vector_db()
    attached_hashes = {f.file_hash for f in files}
    shape = await get_thread_shape(thread_id)
    # Filter out non-dict entries left by older metadata cache formats.
    documents = shape.get("documents", {})
    file_hashes = {k for k, v in documents.items() if isinstance(v, dict)}
    for stale_hash in list(file_hashes):
        if stale_hash not in attached_hashes:
            await remove_document_from_stats(thread_id, stale_hash)

    for file in files:
        file_status = await get_file_status(file.file_hash)
        scoped_status = get_scoped_indexing_status(
            file_status,
            embedding_model=embedding_model,
            thread_id=thread_id,
        )
        chunk_count = await vector_db.get_file_chunk_count(file.file_hash, embedding_model)
        is_ready = ProcessStatus.is_completed(scoped_status.get("status", ProcessStatus.UNKNOWN.value)) or chunk_count > 0
        if not is_ready:
            await remove_document_from_stats(thread_id, file.file_hash)
            continue

        total_chars = int(scoped_status.get("total_chars", 0) or 0)
        await upsert_document_in_stats(
            thread_id,
            file.file_hash,
            {
                "file_name": file.file_name,
                "source_type": file.source_type,
                "chunk_count": chunk_count,
                "total_chars": total_chars,
                "indexing_status": ProcessStatus.COMPLETED.value,
                "indexed_at": scoped_status.get("finished_at"),
            },
        )


# Import at the end to avoid circular dependencies
from app.db import get_file_status
