from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Query

from app.models.requests import (
    MemoryCuratorApplyRequest,
    MemoryCuratorRespondRequest,
    MemorySearchRequest,
)
from app.services.memory_curator_service import (
    MemoryChangedError,
    MemoryCuratorError,
    MemoryCuratorModelUnavailableError,
    MemoryCuratorNotFoundError,
    respond_to_memory_curator,
)
from app.services.memory_tool_service import (
    MemoryToolError,
    MemoryToolNotFoundError,
    apply_confirmed_memory_change,
)
from app.services.effective_memory_service import (
    memory_payload,
    resolve_effective_memory_context,
    serialize_memories_with_relationships,
)
from app.services.memory_review_service import get_memory_review_status
from app.services.memory_service import (
    MemoryVectorCleanupError,
    hard_delete_memory,
    list_scope_memories,
    retry_memory_index,
    search_thread_memory,
)
from app.services.memory_workspace_service import get_memory_workspace_readiness
from app.services.embedding_model_service import (
    EmbeddingModelResolutionError,
    EmbeddingModelUnavailableError,
)


router = APIRouter(tags=["memories"])


def _bad_request_from_value_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@router.post("/memory-workspace/prepare")
async def prepare_memory_workspace_endpoint(
    thread_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
):
    try:
        return await get_memory_workspace_readiness(
            thread_id=thread_id,
            project_id=project_id,
            prepare=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/memory-workspace/status")
async def memory_workspace_status_endpoint(
    thread_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
):
    try:
        return await get_memory_workspace_readiness(
            thread_id=thread_id,
            project_id=project_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _raise_curator_http(exc: Exception):
    if isinstance(exc, MemoryChangedError):
        raise HTTPException(
            status_code=409,
            detail={
                "code": exc.code,
                "message": str(exc),
                "memories": [memory_payload(memory) for memory in exc.memories],
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


def _effective_response(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"memory_records", "excluded_memory_ids"}
    }


@router.get("/memories/effective")
async def global_effective_memories_endpoint(
    limit: int = Query(default=500, ge=1, le=500),
):
    return _effective_response(await resolve_effective_memory_context(limit=limit))


@router.get("/projects/{project_id}/memories/effective")
async def project_effective_memories_endpoint(
    project_id: str,
    limit: int = Query(default=500, ge=1, le=500),
):
    try:
        return _effective_response(await resolve_effective_memory_context(
            project_id=project_id,
            limit=limit,
        ))
    except EmbeddingModelResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/threads/{thread_id}/memories/effective")
async def thread_effective_memories_endpoint(
    thread_id: str,
    limit: int = Query(default=500, ge=1, le=500),
):
    try:
        return _effective_response(await resolve_effective_memory_context(
            thread_id=thread_id,
            limit=limit,
        ))
    except EmbeddingModelResolutionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/threads/{thread_id}/memories/review-status")
async def thread_memory_review_status_endpoint(thread_id: str):
    try:
        return await get_memory_review_status("thread", thread_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


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
        return await apply_confirmed_memory_change(req)
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "embedding_model_unavailable", "message": str(exc)},
        ) from exc
    except MemoryVectorCleanupError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except MemoryToolNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"code": exc.code, "message": str(exc)}) from exc
    except MemoryToolError as exc:
        raise HTTPException(status_code=422, detail={"code": exc.code, "message": str(exc)}) from exc
    except Exception as exc:
        _raise_curator_http(exc)


@router.post("/memories/{memory_id}/index")
async def retry_memory_index_endpoint(
    memory_id: str,
    embedding_model: str | None = Query(default=None),
):
    try:
        memory = await retry_memory_index(memory_id, embedding_model=embedding_model)
    except EmbeddingModelUnavailableError as exc:
        raise HTTPException(status_code=409, detail={"code": "embedding_model_unavailable", "message": str(exc)}) from exc
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    serialized = await serialize_memories_with_relationships([memory])
    return serialized[0]
