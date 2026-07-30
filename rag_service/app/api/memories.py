from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from app.db import (
    MemoryScopeType,
)
from app.models.requests import (
    MemoryCuratorApplyRequest,
    MemoryCuratorRespondRequest,
    MemoryCreateRequest,
    MemorySearchRequest,
)
from app.services.memory_curator_service import (
    MemoryChangedError,
    MemoryCuratorError,
    MemoryCuratorModelUnavailableError,
    MemoryCuratorNotFoundError,
    apply_memory_curator_change_set,
    memory_payload as curator_memory_payload,
    respond_to_memory_curator,
)
from app.services.memory_service import (
    MemoryVectorCleanupError,
    create_and_index_memory,
    hard_delete_memory,
    list_scope_memories,
    retry_memory_index,
    search_thread_memory,
)
from app.services.embedding_model_service import (
    EmbeddingModelResolutionError,
    EmbeddingModelUnavailableError,
)
from app.time_utils import iso_utc_z


router = APIRouter(tags=["memories"])


def _memory_payload(memory) -> Dict[str, Any]:
    return {
        "id": memory.id,
        "scope_type": memory.scope_type,
        "scope_id": memory.scope_id,
        "memory_type": memory.memory_type,
        "content": memory.content,
        "summary": memory.summary,
        "embedding_model": memory.embedding_model,
        "content_hash": memory.content_hash,
        "index_status": memory.index_status,
        "index_attempts": memory.index_attempts,
        "indexed_at": iso_utc_z(memory.indexed_at) if memory.indexed_at else None,
        "index_error": memory.index_error,
        "source_refs_json": memory.source_refs_json if isinstance(memory.source_refs_json, dict) else {},
        "confidence": memory.confidence,
        "status": memory.status,
        "visibility": memory.visibility,
        "created_by": memory.created_by,
        "expires_at": iso_utc_z(memory.expires_at) if memory.expires_at else None,
        "fork_origin_json": memory.fork_origin_json if isinstance(memory.fork_origin_json, dict) else None,
        "created_at": iso_utc_z(memory.created_at) if memory.created_at else None,
        "updated_at": iso_utc_z(memory.updated_at) if memory.updated_at else None,
    }


def _bad_request_from_value_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _raise_curator_http(exc: Exception):
    if isinstance(exc, MemoryChangedError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": exc.code,
                "message": str(exc),
                "memories": [curator_memory_payload(memory) for memory in exc.memories],
            },
        )
    if isinstance(exc, MemoryCuratorModelUnavailableError):
        raise HTTPException(
            status_code=409,
            detail={"code": exc.code, "message": str(exc)},
        )
    if isinstance(exc, MemoryCuratorNotFoundError):
        raise HTTPException(
            status_code=404,
            detail={"code": exc.code, "message": str(exc)},
        )
    if isinstance(exc, MemoryCuratorError):
        raise HTTPException(
            status_code=400,
            detail={"code": exc.code, "message": str(exc)},
        )
    raise exc


@router.get("/memories")
async def list_memories_endpoint(
    scope_type: str = Query(...),
    scope_id: str = Query(...),
    limit: int = Query(default=100, ge=1, le=500),
):
    try:
        rows = await list_scope_memories(scope_type=scope_type, scope_id=scope_id, limit=limit)
    except ValueError as exc:
        raise _bad_request_from_value_error(exc) from exc
    return {"memories": rows}


@router.post("/memories")
async def create_memory_endpoint(req: MemoryCreateRequest):
    if req.scope_type not in {item.value for item in MemoryScopeType}:
        raise HTTPException(status_code=400, detail="Invalid memory scope_type")
    if not req.scope_id.strip() or not req.content.strip():
        raise HTTPException(status_code=400, detail="scope_id and content are required")
    try:
        memory = await create_and_index_memory(
            scope_type=req.scope_type,
            scope_id=req.scope_id,
            memory_type=req.memory_type,
            content=req.content,
            summary=req.summary,
            source_refs_json=req.source_refs_json,
            confidence=req.confidence,
            visibility=req.visibility,
            created_by=req.created_by,
            expires_at=req.expires_at,
        )
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_model_unavailable", "message": str(exc)}) from exc
    except (ValueError, EmbeddingModelResolutionError) as exc:
        raise _bad_request_from_value_error(exc) from exc
    return _memory_payload(memory)


@router.delete("/memories/{memory_id}")
async def delete_memory_endpoint(
    memory_id: str,
):
    try:
        result = await hard_delete_memory(memory_id)
    except MemoryVectorCleanupError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not result["deleted"]:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted", "memory_id": memory_id, "vector_cleanup": result["vector_cleanup"]}


@router.post("/threads/{thread_id}/memories/search")
async def search_thread_memories_endpoint(thread_id: str, req: MemorySearchRequest):
    try:
        return await search_thread_memory(
            thread_id=thread_id,
            query=req.query,
            allowed_scopes=req.allowed_scopes,
            max_results=req.max_results,
        )
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_model_unavailable", "message": str(exc)}) from exc
    except EmbeddingModelResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/memory-curator/respond")
async def memory_curator_respond_endpoint(req: MemoryCuratorRespondRequest):
    try:
        return await respond_to_memory_curator(req)
    except Exception as exc:
        _raise_curator_http(exc)


@router.post("/memory-curator/apply")
async def memory_curator_apply_endpoint(req: MemoryCuratorApplyRequest):
    try:
        return await apply_memory_curator_change_set(req)
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "embedding_model_unavailable", "message": str(exc)},
        ) from exc
    except MemoryVectorCleanupError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        _raise_curator_http(exc)


@router.post("/memories/{memory_id}/index")
async def retry_memory_index_endpoint(memory_id: str):
    try:
        memory = await retry_memory_index(memory_id)
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_model_unavailable", "message": str(exc)}) from exc
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _memory_payload(memory)
