"""Context-wide readiness and repair for the Memory workspace."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any, Dict, List, Tuple

from sqlalchemy import or_
from sqlalchemy.future import select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import MemoryScopeType
from app.db.models_sqlmodel import Memory
from app.db import get_project
from app.models.llm_server_client import check_embedding_model_ready
from app.services.embedding_model_service import (
    GLOBAL_MEMORY_EMBEDDING_MODEL,
    resolve_thread_embedding_context,
)
from app.services.memory_policy import LOCAL_USER_MEMORY_SCOPE_ID
from app.services.memory_representation_service import (
    global_representation_status_for_model,
    reset_stale_global_representation_indexes,
    warm_global_representations_for_model,
)
from app.services.memory_service import index_memory_record
from app.time_utils import utc_now


async def _resolve_context(
    thread_id: str | None,
    project_id: str | None,
) -> Tuple[str, str, List[Tuple[str, str]]]:
    if thread_id:
        context = await resolve_thread_embedding_context(thread_id)
        if project_id and project_id != context.project.id:
            raise ValueError("Thread does not belong to the requested project")
        return "thread", context.embedding_model, [
            (MemoryScopeType.THREAD.value, context.thread.id),
            (MemoryScopeType.PROJECT.value, context.project.id),
            (MemoryScopeType.USER.value, LOCAL_USER_MEMORY_SCOPE_ID),
        ]
    if project_id:
        project = await get_project(project_id)
        if project is None:
            raise ValueError("Project not found")
        return "project", project.embedding_model, [
            (MemoryScopeType.PROJECT.value, project.id),
            (MemoryScopeType.USER.value, LOCAL_USER_MEMORY_SCOPE_ID),
        ]
    return "global", GLOBAL_MEMORY_EMBEDDING_MODEL, [
        (MemoryScopeType.USER.value, LOCAL_USER_MEMORY_SCOPE_ID),
    ]


async def _repair_canonical_memories(memory_ids: List[str]) -> None:
    for memory_id in memory_ids:
        async with async_session_maker() as session:
            memory = await session.get(Memory, memory_id)
        if memory is None or memory.index_status not in {"pending", "failed"}:
            continue
        try:
            await index_memory_record(memory)
        except Exception:
            # The canonical index row records the bounded failure for status polling and Retry.
            pass


async def get_memory_workspace_readiness(
    *,
    thread_id: str | None = None,
    project_id: str | None = None,
    prepare: bool = False,
) -> Dict[str, Any]:
    context_type, embedding_model, scopes = await _resolve_context(thread_id, project_id)
    model_ready = await check_embedding_model_ready(embedding_model)
    filters = [
        (Memory.scope_type == scope_type) & (Memory.scope_id == scope_id)
        for scope_type, scope_id in scopes
        if scope_type != MemoryScopeType.USER.value or embedding_model == GLOBAL_MEMORY_EMBEDDING_MODEL
    ]
    async with async_session_maker() as session:
        async with session.begin():
            memories = list((await session.execute(
                select(Memory).where(or_(*filters)) if filters else select(Memory).where(False)
            )).scalars().all())
            if prepare:
                cutoff = utc_now() - timedelta(minutes=5)
                for memory in memories:
                    if memory.index_status == "indexing" and memory.updated_at < cutoff:
                        memory.index_status = "failed"
                        memory.index_error = "Indexing was interrupted; retry scheduled by Memory workspace readiness."
                        memory.updated_at = utc_now()
    canonical_counts = {"indexed": 0, "pending": 0, "indexing": 0, "failed": 0}
    for memory in memories:
        canonical_counts[memory.index_status] = canonical_counts.get(memory.index_status, 0) + 1
    repair_ids = [
        memory.id for memory in memories
        if memory.index_status in {"pending", "failed"}
    ]
    if prepare:
        await reset_stale_global_representation_indexes(embedding_model)
    representation_status = await global_representation_status_for_model(embedding_model)
    repair_started = bool(model_ready and prepare and (repair_ids or not representation_status["ready"]))
    if repair_started:
        if repair_ids:
            asyncio.create_task(_repair_canonical_memories(repair_ids))
        if not representation_status["ready"]:
            asyncio.create_task(warm_global_representations_for_model(embedding_model))

    canonical_ready = len(memories) == canonical_counts.get("indexed", 0)
    failed_count = canonical_counts.get("failed", 0) + representation_status["failed_count"]
    if not model_ready:
        status = "blocked"
    elif canonical_ready and representation_status["ready"]:
        status = "ready"
    elif failed_count and not repair_started:
        status = "error"
    else:
        status = "indexing"
    return {
        "context_type": context_type,
        "thread_id": thread_id,
        "project_id": project_id,
        "embedding_model": embedding_model,
        "embedding_model_ready": model_ready,
        "status": status,
        "ready": status == "ready",
        "canonical": {
            "total_count": len(memories),
            **canonical_counts,
        },
        "global_representations": representation_status,
    }
