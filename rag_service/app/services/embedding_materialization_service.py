"""Durable, model-aware orchestration for document and memory embeddings."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import timedelta
from typing import Any, Dict, Iterable, List

from sqlalchemy import delete, select

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import MemoryScopeType
from app.db.models_sqlmodel import ChatTurn, EmbeddingJob, GlobalMemoryRepresentation, Memory
from app.db.vector import get_vector_db
from app.services.embedding_model_service import (
    GLOBAL_MEMORY_EMBEDDING_MODEL,
    require_embedding_model_ready,
)
from app.services.memory_policy import LOCAL_USER_MEMORY_SCOPE_ID
from app.time_utils import utc_now


logger = logging.getLogger(__name__)

RESOURCE_DOCUMENT = "document"
RESOURCE_CHAT_MEMORY = "chat_memory"
RESOURCE_GLOBAL_MEMORY = "global_memory"
JOB_PENDING = "pending"
JOB_RUNNING = "running"
JOB_COMPLETED = "completed"
JOB_FAILED = "failed"
STALE_JOB_AFTER_SECONDS = 15 * 60
MAX_JOB_ATTEMPTS = 5

def _wake() -> None:
    # The worker polls durable state. Avoid a process-global asyncio.Event because
    # TestClient and multi-worker deployments may use different event loops.
    return None


async def ensure_embedding_job(
    *,
    resource_type: str,
    resource_id: str,
    scope_id: str,
    embedding_model: str,
    source_version: str,
    session=None,
) -> EmbeddingJob:
    """Create or refresh one durable target without duplicating it."""
    values = {
        "resource_type": str(resource_type),
        "resource_id": str(resource_id),
        "scope_id": str(scope_id),
        "embedding_model": str(embedding_model),
        "source_version": str(source_version),
        "status": JOB_PENDING,
        "attempts": 0,
        "error": None,
        "available_at": utc_now(),
        "claimed_at": None,
        "completed_at": None,
        "updated_at": utc_now(),
    }

    async def apply(active):
        row = (await active.execute(
            select(EmbeddingJob)
            .where(
                EmbeddingJob.resource_type == values["resource_type"],
                EmbeddingJob.resource_id == values["resource_id"],
                EmbeddingJob.scope_id == values["scope_id"],
                EmbeddingJob.embedding_model == values["embedding_model"],
            )
            .with_for_update()
        )).scalar_one_or_none()
        if row is None:
            row = EmbeddingJob(**values)
            active.add(row)
        elif row.source_version != values["source_version"]:
            row.source_version = values["source_version"]
            row.status = JOB_PENDING
            row.attempts = 0
            row.error = None
            row.available_at = utc_now()
            row.claimed_at = None
            row.completed_at = None
            row.updated_at = utc_now()
        elif row.status in {JOB_FAILED, JOB_PENDING}:
            row.updated_at = utc_now()
        await active.flush()
        return row

    if session is not None:
        row = await apply(session)
    else:
        async with async_session_maker() as owned:
            async with owned.begin():
                row = await apply(owned)
    _wake()
    return row


async def ensure_global_representation_job(
    memory: Memory,
    embedding_model: str,
    *,
    session=None,
) -> EmbeddingJob | None:
    model = str(embedding_model or "").strip()
    if not model or model == GLOBAL_MEMORY_EMBEDDING_MODEL:
        return None

    async def apply(active):
        representation = await active.get(GlobalMemoryRepresentation, (memory.id, model))
        if (
            representation is not None
            and representation.index_status == JOB_COMPLETED
            and representation.content_hash == memory.content_hash
        ):
            return None
        if representation is None:
            representation = GlobalMemoryRepresentation(
                memory_id=memory.id,
                embedding_model=model,
                content_hash=memory.content_hash,
                index_status=JOB_PENDING,
                index_attempts=0,
                created_at=utc_now(),
                updated_at=utc_now(),
            )
            active.add(representation)
        elif representation.content_hash != memory.content_hash:
            representation.content_hash = memory.content_hash
            representation.index_status = JOB_PENDING
            representation.indexed_at = None
            representation.index_error = None
            representation.updated_at = utc_now()
        await active.flush()
        return await ensure_embedding_job(
            resource_type=RESOURCE_GLOBAL_MEMORY,
            resource_id=memory.id,
            scope_id=LOCAL_USER_MEMORY_SCOPE_ID,
            embedding_model=model,
            source_version=memory.content_hash,
            session=active,
        )

    if session is not None:
        result = await apply(session)
    else:
        async with async_session_maker() as owned:
            async with owned.begin():
                result = await apply(owned)
    return result


async def enqueue_global_model_jobs(embedding_model: str, *, limit: int | None = None) -> int:
    model = str(embedding_model or "").strip()
    if not model or model == GLOBAL_MEMORY_EMBEDDING_MODEL:
        return 0
    async with async_session_maker() as session:
        async with session.begin():
            query = select(Memory).where(
                Memory.scope_type == MemoryScopeType.USER.value,
                Memory.scope_id == LOCAL_USER_MEMORY_SCOPE_ID,
            ).order_by(Memory.created_at, Memory.id)
            if limit is not None:
                query = query.limit(max(1, int(limit)))
            memories = list((await session.execute(query)).scalars().all())
            for memory in memories:
                await ensure_global_representation_job(memory, model, session=session)
    return len(memories)


async def enqueue_global_jobs_for_active_models(memory: Memory, *, session=None) -> int:
    from app.db.models_sqlmodel import Project

    if memory.scope_type != MemoryScopeType.USER.value or memory.scope_id != LOCAL_USER_MEMORY_SCOPE_ID:
        return 0

    async def apply(active):
        models = list((await active.execute(select(Project.embedding_model).distinct())).scalars().all())
        for model in models:
            await ensure_global_representation_job(memory, str(model), session=active)
        return len(models)

    if session is not None:
        return await apply(session)
    async with async_session_maker() as owned:
        async with owned.begin():
            return await apply(owned)


async def reconcile_thread_embedding_targets(
    thread_id: str,
    embedding_model: str,
    file_hashes: Iterable[str] | None = None,
) -> Dict[str, int]:
    """Enqueue only missing document and Global vectors for one opened thread."""
    from app.db import get_effective_thread_files

    await require_embedding_model_ready(embedding_model)
    documents = await get_effective_thread_files(thread_id)
    if file_hashes:
        wanted = {str(item) for item in file_hashes}
        documents = [file for file in documents if str(file.file_hash) in wanted]
    document_count = 0
    vector_db = get_vector_db()
    for file in documents:
        file_hash = str(file.file_hash)
        if await vector_db.has_file_indexed(thread_id, file_hash, embedding_model):
            continue
        await ensure_embedding_job(
            resource_type=RESOURCE_DOCUMENT,
            resource_id=file_hash,
            scope_id=thread_id,
            embedding_model=embedding_model,
            source_version=file_hash,
        )
        document_count += 1

    global_count = 0
    chat_count = 0
    from app.db import get_thread_turns
    turns = await get_thread_turns(thread_id, limit=10000)
    for turn in turns:
        payload = turn.payload or {}
        answer = str(payload.get("answer") or "").strip()
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        compact_text = str(metadata.get("context_compact") or "").strip()
        if not answer or not compact_text:
            continue
        if await vector_db.has_chat_memory_indexed(thread_id, turn.id):
            continue
        source_version = hashlib.sha256(f"{compact_text}\n{answer}".encode()).hexdigest()
        await ensure_embedding_job(
            resource_type=RESOURCE_CHAT_MEMORY,
            resource_id=turn.id,
            scope_id=thread_id,
            embedding_model=embedding_model,
            source_version=source_version,
        )
        chat_count += 1
    async with async_session_maker() as session:
        memories = list((await session.execute(
            select(Memory).where(
                Memory.scope_type == MemoryScopeType.USER.value,
                Memory.scope_id == LOCAL_USER_MEMORY_SCOPE_ID,
            )
        )).scalars().all())
        for memory in memories:
            representation = await session.get(GlobalMemoryRepresentation, (memory.id, embedding_model))
            if representation and representation.index_status == JOB_COMPLETED and representation.content_hash == memory.content_hash:
                continue
            await ensure_global_representation_job(memory, embedding_model, session=session)
            global_count += 1
    return {"documents": document_count, "chat_memories": chat_count, "global_memories": global_count}


async def claim_embedding_jobs(*, limit: int = 10) -> List[EmbeddingJob]:
    now = utc_now()
    stale_cutoff = now - timedelta(seconds=STALE_JOB_AFTER_SECONDS)
    async with async_session_maker() as session:
        async with session.begin():
            await session.execute(
                EmbeddingJob.__table__.update()
                .where(
                    EmbeddingJob.status == JOB_RUNNING,
                    EmbeddingJob.claimed_at < stale_cutoff,
                )
                .values(status=JOB_FAILED, error="Embedding job was reclaimed after worker restart", available_at=now, updated_at=now)
            )
            rows = list((await session.execute(
                select(EmbeddingJob)
                .where(
                    EmbeddingJob.status.in_((JOB_PENDING, JOB_FAILED)),
                    EmbeddingJob.available_at <= now,
                    EmbeddingJob.attempts < MAX_JOB_ATTEMPTS,
                )
                .order_by(EmbeddingJob.available_at, EmbeddingJob.created_at, EmbeddingJob.id)
                .limit(max(1, int(limit)))
                .with_for_update(skip_locked=True)
            )).scalars().all())
            for row in rows:
                row.status = JOB_RUNNING
                row.attempts = int(row.attempts or 0) + 1
                row.claimed_at = now
                row.error = None
                row.updated_at = now
            await session.flush()
            return rows


async def complete_embedding_job(job: EmbeddingJob) -> None:
    async with async_session_maker() as session:
        async with session.begin():
            row = await session.get(EmbeddingJob, job.id, with_for_update=True)
            if row and row.status == JOB_RUNNING and row.source_version == job.source_version:
                row.status = JOB_COMPLETED
                row.completed_at = utc_now()
                row.claimed_at = None
                row.updated_at = utc_now()


async def fail_embedding_job(job: EmbeddingJob, error: Exception) -> None:
    now = utc_now()
    delay = min(300, 2 ** max(0, int(job.attempts or 1)))
    async with async_session_maker() as session:
        async with session.begin():
            row = await session.get(EmbeddingJob, job.id, with_for_update=True)
            if row and row.status == JOB_RUNNING and row.source_version == job.source_version:
                row.status = JOB_FAILED
                row.error = str(error)[:2000]
                row.claimed_at = None
                row.available_at = now + timedelta(seconds=delay)
                row.updated_at = now


async def process_embedding_job(job: EmbeddingJob) -> None:
    if job.resource_type == RESOURCE_GLOBAL_MEMORY:
        from app.services.memory_representation_service import index_global_representation
        async with async_session_maker() as session:
            memory = await session.get(Memory, job.resource_id)
            if memory is None or memory.content_hash != job.source_version:
                return
        await index_global_representation(job.resource_id, job.embedding_model)
        async with async_session_maker() as session:
            memory = await session.get(Memory, job.resource_id)
            if memory is None or memory.content_hash != job.source_version:
                raise RuntimeError("Global memory changed while embedding job was running")
        return
    if job.resource_type == RESOURCE_DOCUMENT:
        from app.rag.indexer import index_document_for_thread
        result = await index_document_for_thread(
            thread_id=job.scope_id,
            file_hash=job.resource_id,
            embedding_model=job.embedding_model,
        )
        if result.get("status") not in {"success", "completed"}:
            raise RuntimeError(result.get("message", "Document indexing failed"))
        return
    if job.resource_type == RESOURCE_CHAT_MEMORY:
        from app.rag.indexer import index_chat_memory_from_compact_for_thread
        async with async_session_maker() as session:
            turn = await session.get(ChatTurn, job.resource_id)
            if turn is None or turn.thread_id != job.scope_id:
                return
            payload = turn.payload or {}
            answer = str(payload.get("answer") or "").strip()
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            compact_text = str(metadata.get("context_compact") or "").strip()
            current_version = hashlib.sha256(f"{compact_text}\n{answer}".encode()).hexdigest()
            if current_version != job.source_version:
                return
            message_created_at = turn.completed_at or turn.created_at
        result = await index_chat_memory_from_compact_for_thread(
            thread_id=job.scope_id,
            message_id=job.resource_id,
            compact_text=compact_text,
            answer=answer,
            embedding_model=job.embedding_model,
            message_created_at=message_created_at,
        )
        if result.get("status") not in {"success", "completed"}:
            raise RuntimeError(result.get("message", "Chat-memory indexing failed"))
        return
    raise RuntimeError(f"Unsupported embedding job resource type: {job.resource_type}")


async def drain_embedding_jobs(*, limit: int = 10) -> int:
    jobs = await claim_embedding_jobs(limit=limit)
    for job in jobs:
        try:
            await process_embedding_job(job)
            await complete_embedding_job(job)
        except Exception as exc:
            logger.warning("Embedding job failed | id=%s type=%s resource=%s: %s", job.id, job.resource_type, job.resource_id, exc)
            await fail_embedding_job(job, exc)
    return len(jobs)


async def embedding_job_worker(stop_event: asyncio.Event, *, interval: float = 2.0) -> None:
    while not stop_event.is_set():
        processed = await drain_embedding_jobs()
        if processed:
            continue
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def cancel_embedding_jobs_for_memory(memory_id: str, *, session=None) -> None:
    async def apply(active):
        await active.execute(delete(EmbeddingJob).where(
            EmbeddingJob.resource_type == RESOURCE_GLOBAL_MEMORY,
            EmbeddingJob.resource_id == memory_id,
        ))

    if session is not None:
        await apply(session)
    else:
        async with async_session_maker() as owned:
            async with owned.begin():
                await apply(owned)


async def cancel_embedding_jobs_for_scope(scope_id: str, *, session=None) -> None:
    async def apply(active):
        await active.execute(delete(EmbeddingJob).where(EmbeddingJob.scope_id == scope_id))

    if session is not None:
        await apply(session)
    else:
        async with async_session_maker() as owned:
            async with owned.begin():
                await apply(owned)
