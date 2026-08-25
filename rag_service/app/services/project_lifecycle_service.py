"""Project snapshot cloning and hard-deletion lifecycle."""

from __future__ import annotations

import copy
from datetime import datetime
import uuid
from typing import Any, Dict

from sqlalchemy import and_, delete, func, or_
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import AgentRunStatus, ChatTurnStatus, MemoryScopeType
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import (
    AgentRun,
    ChatTurn,
    File,
    Memory,
    MemoryEvent,
    MemoryOverride,
    MemoryReviewState,
    MemoryScopeActivity,
    Project,
    ProjectFile,
    Thread,
    ThreadDocumentAnnotation,
    ThreadFile,
)
from app.db.vector import get_vector_db
from app.services.embedding_model_service import require_embedding_model_ready
from app.services.file_cleanup_service import delete_file_artifacts
from app.services.memory_service import index_memory_record
from app.services.memory_policy import merge_project_settings_json
from app.time_utils import iso_utc_z, utc_now


async def delete_agent_checkpoints(*args: Any, **kwargs: Any):
    from app.runtime.langgraph.checkpointing import delete_agent_checkpoints as implementation
    return await implementation(*args, **kwargs)


ACTIVE_RUN_STATUSES = {
    AgentRunStatus.RUNNING.value,
    AgentRunStatus.AWAITING_HUMAN.value,
}
TERMINAL_RUN_STATUSES = {
    AgentRunStatus.COMPLETED.value,
    AgentRunStatus.CLARIFICATION.value,
    AgentRunStatus.FAILED.value,
    AgentRunStatus.REJECTED.value,
    AgentRunStatus.EXPIRED.value,
    AgentRunStatus.CANCELLED.value,
}


class ProjectLifecycleError(RuntimeError):
    code = "project_lifecycle_error"


class ProjectNotFoundError(ProjectLifecycleError):
    code = "project_not_found"


class ProtectedProjectError(ProjectLifecycleError):
    code = "protected_project"


class ProjectActiveRunsError(ProjectLifecycleError):
    code = "active_agent_runs"


class ProjectCleanupError(ProjectLifecycleError):
    code = "project_cleanup_failed"


def _replace_ids(value: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, list):
        return [_replace_ids(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_ids(item, replacements) for key, item in value.items()}
    return copy.deepcopy(value)


def _valid_debug_trace(run: AgentRun) -> bool:
    debug = run.debug_trace_json
    return (
        run.status in TERMINAL_RUN_STATUSES
        and isinstance(debug, dict)
        and debug.get("version") == 2
        and isinstance(debug.get("trace"), dict)
        and isinstance(debug.get("summary"), dict)
    )


async def _default_project_id() -> str:
    from app.db import ensure_default_project

    return (await ensure_default_project()).id


async def get_project_lifecycle_summary(project_id: str) -> Dict[str, Any]:
    default_project_id = await _default_project_id()
    async with async_session_maker() as session:
        project = await session.get(Project, project_id)
        if project is None:
            raise ProjectNotFoundError("Project not found")

        thread_ids = list((await session.execute(
            select(Thread.id).where(Thread.project_id == project_id)
        )).scalars().all())
        active_runs = 0
        agent_runs = 0
        direct_files = 0
        direct_file_hashes: set[str] = set()
        annotations = 0
        thread_memories = 0
        if thread_ids:
            active_runs = int((await session.execute(
                select(func.count(AgentRun.id)).where(
                    AgentRun.thread_id.in_(thread_ids),
                    AgentRun.status.in_(ACTIVE_RUN_STATUSES),
                )
            )).scalar() or 0)
            agent_runs = int((await session.execute(
                select(func.count(AgentRun.id)).where(AgentRun.thread_id.in_(thread_ids))
            )).scalar() or 0)
            direct_files = int((await session.execute(
                select(func.count()).select_from(ThreadFile).where(ThreadFile.thread_id.in_(thread_ids))
            )).scalar() or 0)
            direct_file_hashes = set((await session.execute(
                select(ThreadFile.file_hash).where(ThreadFile.thread_id.in_(thread_ids))
            )).scalars().all())
            annotations = int((await session.execute(
                select(func.count()).select_from(ThreadDocumentAnnotation).where(
                    ThreadDocumentAnnotation.thread_id.in_(thread_ids)
                )
            )).scalar() or 0)
            thread_memories = int((await session.execute(
                select(func.count(Memory.id)).where(
                    Memory.scope_type == MemoryScopeType.THREAD.value,
                    Memory.scope_id.in_(thread_ids),
                )
            )).scalar() or 0)

        project_files = int((await session.execute(
            select(func.count()).select_from(ProjectFile).where(ProjectFile.project_id == project_id)
        )).scalar() or 0)
        project_file_hashes = set((await session.execute(
            select(ProjectFile.file_hash).where(ProjectFile.project_id == project_id)
        )).scalars().all())
        affected_file_hashes = project_file_hashes | direct_file_hashes
        outside_refs = await _outside_file_references(
            session,
            project_id=project_id,
            thread_ids=thread_ids,
            file_hashes=affected_file_hashes,
        )
        shared_files = sum(1 for refs in outside_refs.values() if refs["any"])
        project_memories = int((await session.execute(
            select(func.count(Memory.id)).where(
                Memory.scope_type == MemoryScopeType.PROJECT.value,
                Memory.scope_id == project_id,
            )
        )).scalar() or 0)
        scoped_memory_ids = list((await session.execute(
            select(Memory.id).where(or_(
                and_(
                    Memory.scope_type == MemoryScopeType.PROJECT.value,
                    Memory.scope_id == project_id,
                ),
                and_(
                    Memory.scope_type == MemoryScopeType.THREAD.value,
                    Memory.scope_id.in_(thread_ids),
                ) if thread_ids else False,
            ))
        )).scalars().all())
        override_count = int((await session.execute(
            select(func.count()).select_from(MemoryOverride).where(or_(
                MemoryOverride.overriding_memory_id.in_(scoped_memory_ids),
                MemoryOverride.overridden_memory_id.in_(scoped_memory_ids),
            ))
        )).scalar() or 0) if scoped_memory_ids else 0
    protected = project_id == default_project_id
    return {
        "project_id": project_id,
        "thread_count": len(thread_ids),
        "project_file_count": project_files,
        "direct_file_count": direct_files,
        "unique_file_count": len(affected_file_hashes),
        "shared_file_count": shared_files,
        "orphan_file_count": len(affected_file_hashes) - shared_files,
        "memory_count": project_memories + thread_memories,
        "project_memory_count": project_memories,
        "thread_memory_count": thread_memories,
        "memory_override_count": override_count,
        "annotation_count": annotations,
        "agent_run_count": agent_runs,
        "active_run_count": active_runs,
        "protected": protected,
        "can_delete": not protected and active_runs == 0,
        "can_clone": active_runs == 0,
        "blocked_reason": (
            "protected_project" if protected
            else "active_agent_runs" if active_runs
            else None
        ),
    }


async def clone_project(
    project_id: str,
    *,
    name: str,
    include_threads: bool,
) -> Dict[str, Any]:
    clone_name = str(name or "").strip()
    if not clone_name:
        raise ValueError("Project name is required")

    await get_project_lifecycle_summary(project_id)
    async with async_session_maker() as session:
        source_project = await session.get(Project, project_id)
        if source_project is None:
            raise ProjectNotFoundError("Project not found")
        active_count = int((await session.execute(
            select(func.count(AgentRun.id))
            .join(Thread, Thread.id == AgentRun.thread_id)
            .where(
                Thread.project_id == project_id,
                AgentRun.status.in_(ACTIVE_RUN_STATUSES),
            )
        )).scalar() or 0)
        if active_count:
            raise ProjectActiveRunsError("Project has active or awaiting-human agent runs")
        embedding_model = source_project.embedding_model

    await require_embedding_model_ready(embedding_model)
    clone_time = utc_now()
    new_project_id = str(uuid.uuid4())
    memories_to_index: list[Memory] = []
    counts = {
        "project_files": 0,
        "project_memories": 0,
        "threads": 0,
        "turns": 0,
        "agent_runs": 0,
        "direct_files": 0,
        "annotations": 0,
        "thread_memories": 0,
        "memory_overrides": 0,
    }

    async with async_session_maker() as session:
        async with session.begin():
            memory_id_map: dict[str, str] = {}
            source_project = await session.get(Project, project_id, with_for_update=True)
            if source_project is None:
                raise ProjectNotFoundError("Project not found")
            active_count = int((await session.execute(
                select(func.count(AgentRun.id))
                .join(Thread, Thread.id == AgentRun.thread_id)
                .where(
                    Thread.project_id == project_id,
                    AgentRun.status.in_(ACTIVE_RUN_STATUSES),
                )
            )).scalar() or 0)
            if active_count:
                raise ProjectActiveRunsError("Project has active or awaiting-human agent runs")

            settings = merge_project_settings_json({}, copy.deepcopy(source_project.settings_json or {}))
            lifecycle = dict(settings.get("lifecycle") or {})
            lifecycle["clone_origin"] = {
                "source_project_id": source_project.id,
                "cloned_at": iso_utc_z(clone_time),
                "include_threads": include_threads,
            }
            settings["lifecycle"] = lifecycle
            cloned_project = Project(
                id=new_project_id,
                name=clone_name,
                description=source_project.description,
                embedding_model=source_project.embedding_model,
                settings_json=settings,
                created_at=clone_time,
            )
            session.add(cloned_project)
            await session.flush()

            source_project_files = list((await session.execute(
                select(ProjectFile).where(ProjectFile.project_id == project_id)
            )).scalars().all())
            for row in source_project_files:
                session.add(ProjectFile(
                    project_id=new_project_id,
                    file_hash=row.file_hash,
                    added_at=row.added_at,
                ))
            counts["project_files"] = len(source_project_files)

            project_memories = list((await session.execute(
                select(Memory).where(
                    Memory.scope_type == MemoryScopeType.PROJECT.value,
                    Memory.scope_id == project_id,
                )
            )).scalars().all())
            for source_memory in project_memories:
                cloned_memory = _clone_memory(
                    source_memory,
                    scope_id=new_project_id,
                    source_project_id=project_id,
                    cloned_at=clone_time,
                )
                session.add(cloned_memory)
                await session.flush()
                session.add(_clone_memory_event(cloned_memory, source_memory, clone_time))
                memories_to_index.append(cloned_memory)
                memory_id_map[source_memory.id] = cloned_memory.id
            counts["project_memories"] = len(project_memories)

            if include_threads:
                source_threads = list((await session.execute(
                    select(Thread).where(Thread.project_id == project_id).order_by(Thread.created_at, Thread.id)
                )).scalars().all())
                for source_thread in source_threads:
                    thread_counts, thread_memories, thread_memory_map = await _clone_thread(
                        session,
                        source_project_id=project_id,
                        target_project_id=new_project_id,
                        source_thread=source_thread,
                        cloned_at=clone_time,
                    )
                    for key, value in thread_counts.items():
                        counts[key] += value
                    memories_to_index.extend(thread_memories)
                    memory_id_map.update(thread_memory_map)
                counts["threads"] = len(source_threads)

            if memory_id_map:
                override_rows = list((await session.execute(
                    select(MemoryOverride).where(
                        MemoryOverride.overriding_memory_id.in_(list(memory_id_map))
                    )
                )).scalars().all())
                target_ids = [row.overridden_memory_id for row in override_rows]
                target_rows = list((await session.execute(
                    select(Memory).where(Memory.id.in_(target_ids))
                )).scalars().all()) if target_ids else []
                target_by_id = {memory.id: memory for memory in target_rows}
                for row in override_rows:
                    target_id = memory_id_map.get(row.overridden_memory_id)
                    if target_id is None:
                        target = target_by_id.get(row.overridden_memory_id)
                        if target and target.scope_type == MemoryScopeType.USER.value:
                            target_id = target.id
                    if target_id:
                        session.add(MemoryOverride(
                            overriding_memory_id=memory_id_map[row.overriding_memory_id],
                            overridden_memory_id=target_id,
                            created_at=clone_time,
                        ))
                        counts["memory_overrides"] += 1

    warnings = []
    for memory in memories_to_index:
        try:
            await index_memory_record(memory)
        except Exception as exc:
            warnings.append({
                "code": "memory_index_failed",
                "memory_id": memory.id,
                "message": str(exc)[:500],
            })

    async with async_session_maker() as session:
        cloned_project = await session.get(Project, new_project_id)
    return {"project": cloned_project, "counts": counts, "warnings": warnings}


def _clone_memory(
    source: Memory,
    *,
    scope_id: str,
    source_project_id: str,
    cloned_at: datetime,
) -> Memory:
    fork_origin = {
        "source_memory_id": source.id,
        "source_project_id": source_project_id,
        "copy_mode": "project_clone_snapshot",
        "cloned_at": iso_utc_z(cloned_at),
    }
    return Memory(
        id=str(uuid.uuid4()),
        scope_type=source.scope_type,
        scope_id=scope_id,
        content=source.content,
        embedding_model=source.embedding_model,
        content_hash=source.content_hash,
        index_status="pending",
        index_attempts=0,
        indexed_at=None,
        index_error=None,
        source_refs_json={
            **copy.deepcopy(source.source_refs_json or {}),
            "fork_origin": fork_origin,
        },
        attributes_json=copy.deepcopy(source.attributes_json or {}),
        created_at=cloned_at,
        updated_at=cloned_at,
    )


def _clone_memory_event(
    cloned: Memory,
    source: Memory,
    cloned_at: datetime,
) -> MemoryEvent:
    return MemoryEvent(
        memory_id=cloned.id,
        event_type="project_clone_snapshot",
        actor_id="system",
        payload_json={"source_memory_id": source.id},
        created_at=cloned_at,
    )


async def _clone_thread(
    session,
    *,
    source_project_id: str,
    target_project_id: str,
    source_thread: Thread,
    cloned_at: datetime,
) -> tuple[Dict[str, int], list[Memory], dict[str, str]]:
    new_thread_id = str(uuid.uuid4())
    metadata = copy.deepcopy(source_thread.thread_metadata or {})
    metadata.pop("fork_children", None)
    metadata["project_clone"] = {
        "source_project_id": source_project_id,
        "source_thread_id": source_thread.id,
        "cloned_at": iso_utc_z(cloned_at),
    }
    cloned_thread = Thread(
        id=new_thread_id,
        project_id=target_project_id,
        name=source_thread.name,
        embedding_model=source_thread.embedding_model,
        settings=copy.deepcopy(source_thread.settings or {}),
        thread_metadata=metadata,
        total_qa_pairs=source_thread.total_qa_pairs,
        total_qa_chars=source_thread.total_qa_chars,
        avg_qa_chars=source_thread.avg_qa_chars,
        last_qa_at=source_thread.last_qa_at,
        documents_meta=copy.deepcopy(source_thread.documents_meta or {}),
        stats_last_updated_at=source_thread.stats_last_updated_at,
        created_at=source_thread.created_at,
        updated_at=source_thread.updated_at,
    )
    session.add(cloned_thread)
    await session.flush()

    direct_files = list((await session.execute(
        select(ThreadFile).where(ThreadFile.thread_id == source_thread.id)
    )).scalars().all())
    for row in direct_files:
        session.add(ThreadFile(
            thread_id=new_thread_id,
            file_hash=row.file_hash,
            added_at=row.added_at,
            annotations=copy.deepcopy(row.annotations or []),
            annotations_updated_at=row.annotations_updated_at,
        ))

    annotations = list((await session.execute(
        select(ThreadDocumentAnnotation).where(
            ThreadDocumentAnnotation.thread_id == source_thread.id
        )
    )).scalars().all())
    for row in annotations:
        session.add(ThreadDocumentAnnotation(
            thread_id=new_thread_id,
            file_hash=row.file_hash,
            annotations=copy.deepcopy(row.annotations or []),
            created_at=row.created_at,
            updated_at=row.updated_at,
        ))

    source_turns = list((await session.execute(
        select(ChatTurn).where(
            ChatTurn.thread_id == source_thread.id,
            ChatTurn.status == ChatTurnStatus.COMPLETED.value,
        ).order_by(ChatTurn.created_at, ChatTurn.id)
    )).scalars().all())
    turn_map = {turn.id: str(uuid.uuid4()) for turn in source_turns}
    curator_metadata = metadata.get("memory_curator")
    if isinstance(curator_metadata, dict):
        mapped_cursor_id = turn_map.get(str(curator_metadata.get("reviewed_through_turn_id") or ""))
        if mapped_cursor_id:
            metadata["memory_curator"] = {
                **curator_metadata,
                "reviewed_through_turn_id": mapped_cursor_id,
            }
        else:
            metadata.pop("memory_curator", None)
        replace_jsonb_field(cloned_thread, "thread_metadata", metadata)
    source_run_ids = {turn.agent_run_id for turn in source_turns if turn.agent_run_id}
    source_runs = []
    if source_run_ids:
        source_runs = list((await session.execute(
            select(AgentRun).where(AgentRun.id.in_(source_run_ids))
        )).scalars().all())
    run_map = {
        run.id: str(uuid.uuid4())
        for run in source_runs
        if _valid_debug_trace(run)
    }
    replacements = {
        source_project_id: target_project_id,
        source_thread.id: new_thread_id,
        **turn_map,
        **run_map,
    }
    for run in source_runs:
        new_run_id = run_map.get(run.id)
        if not new_run_id:
            continue
        run_metadata = copy.deepcopy(run.run_metadata_json or {})
        run_metadata.pop("builder_session_id", None)
        run_metadata.update({
            "historical_clone": True,
            "source_project_id": source_project_id,
            "source_thread_id": source_thread.id,
            "source_run_id": run.id,
            "cloned_at": iso_utc_z(cloned_at),
        })
        session.add(AgentRun(
            id=new_run_id,
            thread_id=new_thread_id,
            user_id=run.user_id,
            workflow_id=run.workflow_id,
            run_metadata_json=run_metadata,
            resolved_spec_json=copy.deepcopy(run.resolved_spec_json or {}),
            status=run.status,
            checkpoint_thread_id=None,
            pending_interrupt_json=None,
            started_at=run.started_at,
            completed_at=run.completed_at,
            error_json=copy.deepcopy(run.error_json),
            metrics_json=copy.deepcopy(run.metrics_json or {}),
            debug_trace_json=_replace_ids(run.debug_trace_json, replacements),
        ))
    await session.flush()

    for turn in source_turns:
        mapped_run_id = run_map.get(turn.agent_run_id or "")
        session.add(ChatTurn(
            id=turn_map[turn.id],
            thread_id=new_thread_id,
            agent_run_id=mapped_run_id,
            agent_run_turn_kind=turn.agent_run_turn_kind if mapped_run_id else None,
            agent_run_sequence=turn.agent_run_sequence if mapped_run_id else None,
            agent_trace_refs_json=(
                _replace_ids(turn.agent_trace_refs_json, replacements)
                if mapped_run_id else None
            ),
            status=turn.status,
            payload=_replace_ids(turn.payload or {}, replacements),
            created_at=turn.created_at,
            updated_at=turn.updated_at,
            completed_at=turn.completed_at,
        ))

    source_memories = list((await session.execute(
        select(Memory).where(
            Memory.scope_type == MemoryScopeType.THREAD.value,
            Memory.scope_id == source_thread.id,
        )
    )).scalars().all())
    cloned_memories = []
    memory_id_map: dict[str, str] = {}
    for source_memory in source_memories:
        cloned_memory = _clone_memory(
            source_memory,
            scope_id=new_thread_id,
            source_project_id=source_project_id,
            cloned_at=cloned_at,
        )
        session.add(cloned_memory)
        await session.flush()
        session.add(_clone_memory_event(cloned_memory, source_memory, cloned_at))
        cloned_memories.append(cloned_memory)
        memory_id_map[source_memory.id] = cloned_memory.id

    return ({
        "turns": len(source_turns),
        "agent_runs": len(run_map),
        "direct_files": len(direct_files),
        "annotations": len(annotations),
        "thread_memories": len(source_memories),
    }, cloned_memories, memory_id_map)


async def delete_project(project_id: str) -> Dict[str, Any]:
    summary = await get_project_lifecycle_summary(project_id)
    if summary["protected"]:
        raise ProtectedProjectError("The default project cannot be deleted")
    if summary["active_run_count"]:
        raise ProjectActiveRunsError("Project has active or awaiting-human agent runs")

    async with async_session_maker() as session:
        project = await session.get(Project, project_id)
        if project is None:
            raise ProjectNotFoundError("Project not found")
        deleted_project_model = project.embedding_model
        thread_ids = list((await session.execute(
            select(Thread.id).where(Thread.project_id == project_id)
        )).scalars().all())
        memories = list((await session.execute(
            select(Memory).where(or_(
                (
                    (Memory.scope_type == MemoryScopeType.PROJECT.value)
                    & (Memory.scope_id == project_id)
                ),
                (
                    (Memory.scope_type == MemoryScopeType.THREAD.value)
                    & Memory.scope_id.in_(thread_ids)
                ) if thread_ids else False,
            ))
        )).scalars().all())
        checkpoint_ids = []
        if thread_ids:
            checkpoint_ids = list((await session.execute(
                select(AgentRun.checkpoint_thread_id).where(
                    AgentRun.thread_id.in_(thread_ids),
                    AgentRun.framework == "langgraph",
                    AgentRun.checkpoint_thread_id.is_not(None),
                )
            )).scalars().all())
        affected_files = set((await session.execute(
            select(ProjectFile.file_hash).where(ProjectFile.project_id == project_id)
        )).scalars().all())
        if thread_ids:
            affected_files.update((await session.execute(
                select(ThreadFile.file_hash).where(ThreadFile.thread_id.in_(thread_ids))
            )).scalars().all())
        files = {}
        if affected_files:
            files = {
                row.file_hash: row
                for row in (await session.execute(
                    select(File).where(File.file_hash.in_(affected_files))
                )).scalars().all()
            }
        outside_file_refs = await _outside_file_references(
            session,
            project_id=project_id,
            thread_ids=thread_ids,
            file_hashes=affected_files,
        )

    from app.services.task_artifact_service import delete_task_resources_for_threads
    await delete_task_resources_for_threads(thread_ids)

    vector_db = get_vector_db()
    for thread_id in thread_ids:
        if not await vector_db.delete_thread_data(thread_id):
            raise ProjectCleanupError(f"Failed to delete vector data for thread {thread_id}")
    memory_scopes = {(memory.scope_type, memory.scope_id, memory.embedding_model) for memory in memories}
    for scope_type, scope_id, model in memory_scopes:
        if not await vector_db.delete_memory_vectors_for_scope(scope_type, scope_id, model):
            raise ProjectCleanupError(f"Failed to delete memory vectors for {scope_type}:{scope_id}")
    if checkpoint_ids:
        try:
            await delete_agent_checkpoints(checkpoint_ids)
        except Exception as exc:
            raise ProjectCleanupError("Failed to delete project checkpoints") from exc

    vector_models_deleted: set[tuple[str, str]] = set()
    orphan_file_hashes = {
        file_hash for file_hash, refs in outside_file_refs.items() if not refs["any"]
    }
    for file_hash in affected_files:
        refs = outside_file_refs[file_hash]
        models = set()
        if project.embedding_model not in refs["models"]:
            models.add(project.embedding_model)
        if file_hash in orphan_file_hashes:
            file_status = (files.get(file_hash).file_status if files.get(file_hash) else {}) or {}
            indexed_models = (file_status.get("indexing_status") or {}).get("models") or {}
            models.update(str(model) for model in indexed_models if model)
        for model in models:
            if not await vector_db.delete_document_vectors_by_file_hash_and_model(file_hash, model):
                raise ProjectCleanupError(
                    f"Failed to delete document vectors for {file_hash} ({model})"
                )
            vector_models_deleted.add((file_hash, model))

    async with async_session_maker() as session:
        async with session.begin():
            project = await session.get(Project, project_id, with_for_update=True)
            if project is None:
                raise ProjectNotFoundError("Project not found")
            active_count = int((await session.execute(
                select(func.count(AgentRun.id))
                .join(Thread, Thread.id == AgentRun.thread_id)
                .where(
                    Thread.project_id == project_id,
                    AgentRun.status.in_(ACTIVE_RUN_STATUSES),
                )
            )).scalar() or 0)
            if active_count:
                raise ProjectActiveRunsError("Project has active or awaiting-human agent runs")

            current_thread_ids = list((await session.execute(
                select(Thread.id).where(Thread.project_id == project_id)
            )).scalars().all())
            memory_ids = list((await session.execute(
                select(Memory.id).where(or_(
                    (
                        (Memory.scope_type == MemoryScopeType.PROJECT.value)
                        & (Memory.scope_id == project_id)
                    ),
                    (
                        (Memory.scope_type == MemoryScopeType.THREAD.value)
                        & Memory.scope_id.in_(current_thread_ids)
                    ) if current_thread_ids else False,
                ))
            )).scalars().all())
            if memory_ids:
                await session.execute(delete(MemoryEvent).where(MemoryEvent.memory_id.in_(memory_ids)))
                await session.execute(delete(Memory).where(Memory.id.in_(memory_ids)))
            if current_thread_ids:
                await session.execute(delete(MemoryReviewState).where(
                    MemoryReviewState.context_type == "thread",
                    MemoryReviewState.context_id.in_(current_thread_ids),
                ))
                await session.execute(delete(MemoryScopeActivity).where(
                    MemoryScopeActivity.scope_type == MemoryScopeType.THREAD.value,
                    MemoryScopeActivity.scope_id.in_(current_thread_ids),
                ))
                await session.execute(delete(Thread).where(Thread.id.in_(current_thread_ids)))
            await session.execute(delete(MemoryReviewState).where(
                MemoryReviewState.context_type == "project",
                MemoryReviewState.context_id == project_id,
            ))
            await session.execute(delete(MemoryScopeActivity).where(
                MemoryScopeActivity.scope_type == MemoryScopeType.PROJECT.value,
                MemoryScopeActivity.scope_id == project_id,
            ))
            await session.execute(delete(ProjectFile).where(ProjectFile.project_id == project_id))
            await session.flush()
            for file_hash in orphan_file_hashes:
                file = await session.get(File, file_hash)
                if file is not None:
                    await session.delete(file)
            await session.delete(project)

    cleanup_warnings = []
    try:
        from app.services.memory_representation_service import cleanup_unused_global_representation_model
        await cleanup_unused_global_representation_model(deleted_project_model)
    except Exception as exc:
        cleanup_warnings.append(str(exc)[:500])
    for file_hash in orphan_file_hashes:
        try:
            await delete_file_artifacts(file_hash)
        except Exception as exc:
            cleanup_warnings.append(str(exc)[:500])

    return {
        "project_id": project_id,
        "deleted": True,
        "counts": {
            **summary,
            "checkpoint_count": len(checkpoint_ids),
            "canonical_files_deleted": len(orphan_file_hashes),
            "document_vector_models_deleted": len(vector_models_deleted),
        },
        "warnings": cleanup_warnings,
    }


async def _outside_file_references(
    session,
    *,
    project_id: str,
    thread_ids: list[str],
    file_hashes: set[str],
) -> Dict[str, Dict[str, Any]]:
    result = {file_hash: {"any": False, "models": set()} for file_hash in file_hashes}
    if not file_hashes:
        return result
    project_rows = (await session.execute(
        select(ProjectFile.file_hash, Project.embedding_model)
        .join(Project, Project.id == ProjectFile.project_id)
        .where(
            ProjectFile.file_hash.in_(file_hashes),
            ProjectFile.project_id != project_id,
        )
    )).all()
    thread_query = (
        select(ThreadFile.file_hash, Thread.embedding_model)
        .join(Thread, Thread.id == ThreadFile.thread_id)
        .where(ThreadFile.file_hash.in_(file_hashes))
    )
    if thread_ids:
        thread_query = thread_query.where(ThreadFile.thread_id.not_in(thread_ids))
    thread_rows = (await session.execute(thread_query)).all()
    for file_hash, model in [*project_rows, *thread_rows]:
        result[file_hash]["any"] = True
        result[file_hash]["models"].add(model)
    return result
