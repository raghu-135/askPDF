"""Secondary model-aware vector representations for canonical Global memory."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from sqlalchemy import and_, delete, func
from sqlalchemy.future import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.connection_sqlmodel import async_session_maker
from app.db.enums import MemoryScopeType
from app.db.models_sqlmodel import GlobalMemoryRepresentation, Memory, Project
from app.db.vector import get_vector_db
from app.models.llm_server_client import get_embedding_model
from app.models.retry import invoke_with_retry
from app.services.embedding_model_service import GLOBAL_MEMORY_EMBEDDING_MODEL, require_embedding_model_ready
from app.services.memory_policy import LOCAL_USER_MEMORY_SCOPE_ID
from app.time_utils import iso_utc_z, utc_now


logger = logging.getLogger(__name__)


def representation_payload(row: GlobalMemoryRepresentation) -> Dict[str, Any]:
    return {
        "embedding_model": row.embedding_model,
        "primary": False,
        "index_status": row.index_status,
        "index_attempts": row.index_attempts,
        "indexed_at": iso_utc_z(row.indexed_at) if row.indexed_at else None,
        "index_error": row.index_error,
    }


async def ensure_global_representations_for_model(
    embedding_model: str,
    *,
    session: AsyncSession | None = None,
) -> List[GlobalMemoryRepresentation]:
    """Idempotently create pending secondary rows for one project model."""

    model = str(embedding_model or "").strip()
    if not model or model == GLOBAL_MEMORY_EMBEDDING_MODEL:
        return []

    async def apply(active: AsyncSession) -> List[GlobalMemoryRepresentation]:
        memories = list((await active.execute(
            select(Memory).where(
                Memory.scope_type == MemoryScopeType.USER.value,
                Memory.scope_id == LOCAL_USER_MEMORY_SCOPE_ID,
            )
        )).scalars().all())
        if memories:
            await active.execute(pg_insert(GlobalMemoryRepresentation).values([
                {
                    "memory_id": memory.id,
                    "embedding_model": model,
                    "content_hash": memory.content_hash,
                    "index_status": "pending",
                    "index_attempts": 0,
                    "created_at": utc_now(),
                    "updated_at": utc_now(),
                }
                for memory in memories
            ]).on_conflict_do_nothing())
            await active.flush()
        return list((await active.execute(
            select(GlobalMemoryRepresentation).where(
                GlobalMemoryRepresentation.embedding_model == model,
            )
        )).scalars().all())

    if session is not None:
        return await apply(session)
    async with async_session_maker() as owned:
        async with owned.begin():
            return await apply(owned)


async def invalidate_global_representations(
    memory: Memory,
    *,
    session: AsyncSession,
) -> None:
    """Create/reset secondary rows after a canonical Global content change."""

    if memory.scope_type != MemoryScopeType.USER.value or memory.scope_id != LOCAL_USER_MEMORY_SCOPE_ID:
        return
    models = list((await session.execute(select(Project.embedding_model).distinct())).scalars().all())
    now = utc_now()
    for model in models:
        if model == GLOBAL_MEMORY_EMBEDDING_MODEL:
            continue
        stmt = pg_insert(GlobalMemoryRepresentation).values(
            memory_id=memory.id,
            embedding_model=model,
            content_hash=memory.content_hash,
            index_status="pending",
            index_attempts=0,
            indexed_at=None,
            index_error=None,
            created_at=now,
            updated_at=now,
        ).on_conflict_do_update(
            index_elements=["memory_id", "embedding_model"],
            set_={
                "content_hash": memory.content_hash,
                "index_status": "pending",
                "indexed_at": None,
                "index_error": None,
                "updated_at": now,
            },
        )
        await session.execute(stmt)


async def index_global_representation(memory_id: str, embedding_model: str) -> int:
    model = str(embedding_model or "").strip()
    if model == GLOBAL_MEMORY_EMBEDDING_MODEL:
        raise ValueError("The primary Global representation uses canonical indexing")
    async with async_session_maker() as session:
        async with session.begin():
            memory = await session.get(Memory, memory_id)
            row = await session.get(GlobalMemoryRepresentation, (memory_id, model))
            if memory is None or row is None:
                raise ValueError("Global memory representation not found")
            if memory.scope_type != MemoryScopeType.USER.value or memory.scope_id != LOCAL_USER_MEMORY_SCOPE_ID:
                raise ValueError("Secondary representations are limited to Global memory")
            row.index_status = "indexing"
            row.index_attempts = int(row.index_attempts or 0) + 1
            row.index_error = None
            row.updated_at = utc_now()
            content = memory.content
            content_hash = memory.content_hash
            source_refs = memory.source_refs_json or {}
            created_at = memory.created_at
            updated_at = memory.updated_at
    try:
        await require_embedding_model_ready(model)
        vector = await invoke_with_retry(get_embedding_model(model).aembed_query, content)
        inserted = await get_vector_db().index_memory(
            memory_id=memory_id,
            scope_type=MemoryScopeType.USER.value,
            scope_id=LOCAL_USER_MEMORY_SCOPE_ID,
            content=content,
            metadata={"source_refs": source_refs, "content_hash": content_hash, "secondary": True},
            created_at=iso_utc_z(created_at) if created_at else None,
            updated_at=iso_utc_z(updated_at) if updated_at else None,
            embedding=vector,
            embedding_model=model,
        )
        async with async_session_maker() as session:
            async with session.begin():
                row = await session.get(GlobalMemoryRepresentation, (memory_id, model))
                if row:
                    row.index_status = "indexed"
                    row.content_hash = content_hash
                    row.indexed_at = utc_now()
                    row.index_error = None
                    row.updated_at = utc_now()
        return inserted
    except Exception as exc:
        async with async_session_maker() as session:
            async with session.begin():
                row = await session.get(GlobalMemoryRepresentation, (memory_id, model))
                if row:
                    row.index_status = "failed"
                    row.index_error = str(exc)[:2000]
                    row.updated_at = utc_now()
        raise


async def warm_global_representations_for_model(embedding_model: str, *, limit: int = 100) -> Dict[str, Any]:
    await ensure_global_representations_for_model(embedding_model)
    async with async_session_maker() as session:
        rows = list((await session.execute(
            select(GlobalMemoryRepresentation)
            .where(
                GlobalMemoryRepresentation.embedding_model == embedding_model,
                GlobalMemoryRepresentation.index_status.in_(("pending", "failed")),
            )
            .order_by(GlobalMemoryRepresentation.updated_at)
            .limit(max(1, min(limit, 500)))
        )).scalars().all())
    indexed, failed = [], []
    for row in rows:
        try:
            await index_global_representation(row.memory_id, row.embedding_model)
            indexed.append(row.memory_id)
        except Exception:
            failed.append(row.memory_id)
    return {"embedding_model": embedding_model, "indexed_ids": indexed, "failed_ids": failed}


async def list_memory_representations(
    memory_ids: List[str],
    *,
    session: AsyncSession | None = None,
) -> Dict[str, List[Dict[str, Any]]]:
    if not memory_ids:
        return {}
    if session is not None:
        rows = list((await session.execute(
            select(GlobalMemoryRepresentation).where(GlobalMemoryRepresentation.memory_id.in_(memory_ids))
        )).scalars().all())
    else:
        async with async_session_maker() as owned_session:
            rows = list((await owned_session.execute(
                select(GlobalMemoryRepresentation).where(GlobalMemoryRepresentation.memory_id.in_(memory_ids))
            )).scalars().all())
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.memory_id, []).append(representation_payload(row))
    return grouped


async def cleanup_unused_global_representation_model(embedding_model: str) -> bool:
    if embedding_model == GLOBAL_MEMORY_EMBEDDING_MODEL:
        return False
    async with async_session_maker() as session:
        project_count = int((await session.execute(
            select(func.count(Project.id)).where(Project.embedding_model == embedding_model)
        )).scalar() or 0)
        if project_count:
            return False
        rows = list((await session.execute(
            select(GlobalMemoryRepresentation).where(GlobalMemoryRepresentation.embedding_model == embedding_model)
        )).scalars().all())
    for row in rows:
        if not await get_vector_db().delete_memory_vectors(row.memory_id, embedding_model):
            raise RuntimeError(f"Failed to delete Global memory representation {row.memory_id}:{embedding_model}")
    async with async_session_maker() as session:
        async with session.begin():
            await session.execute(delete(GlobalMemoryRepresentation).where(
                GlobalMemoryRepresentation.embedding_model == embedding_model
            ))
    return bool(rows)
