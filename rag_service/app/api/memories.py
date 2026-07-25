from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query

from app.db import (
    MemoryCandidateStatus,
    MemoryScopeType,
    MemoryStatus,
    create_memory_candidate,
    get_memory_repo,
    list_memory_candidates,
    update_memory_status,
)
from app.models.requests import (
    MemoryCandidateCreateRequest,
    MemoryCandidateResolveRequest,
    MemoryCreateRequest,
    MemorySearchRequest,
    MemoryStatusUpdateRequest,
)
from app.services.memory_service import (
    MemoryVectorCleanupError,
    create_and_index_memory,
    hard_delete_memory,
    hard_delete_memory_candidate,
    list_scope_memories,
    search_thread_memory,
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


def _candidate_payload(candidate) -> Dict[str, Any]:
    return {
        "id": candidate.id,
        "source_thread_id": candidate.source_thread_id,
        "source_project_id": candidate.source_project_id,
        "source_agent_run_id": candidate.source_agent_run_id,
        "source_turn_id": candidate.source_turn_id,
        "proposed_scope_type": candidate.proposed_scope_type,
        "proposed_scope_id": candidate.proposed_scope_id,
        "memory_type": candidate.memory_type,
        "content": candidate.content,
        "confidence": candidate.confidence,
        "reason": candidate.reason,
        "status": candidate.status,
        "created_by": candidate.created_by,
        "created_at": iso_utc_z(candidate.created_at) if candidate.created_at else None,
        "updated_at": iso_utc_z(candidate.updated_at) if candidate.updated_at else None,
    }


def _bad_request_from_value_error(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


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
            embedding_model=req.embedding_model,
            scope_type=req.scope_type,
            scope_id=req.scope_id,
            memory_type=req.memory_type,
            content=req.content,
            summary=req.summary,
            source_refs_json=req.source_refs_json,
            confidence=req.confidence,
            visibility=req.visibility,
            created_by=req.created_by,
        )
    except ValueError as exc:
        raise _bad_request_from_value_error(exc) from exc
    return _memory_payload(memory)


@router.put("/memories/{memory_id}/status")
async def update_memory_status_endpoint(memory_id: str, req: MemoryStatusUpdateRequest):
    if req.status not in {item.value for item in MemoryStatus}:
        raise HTTPException(status_code=400, detail="Invalid memory status")
    try:
        memory = await update_memory_status(
            memory_id,
            status=req.status,
            actor_id=req.actor_id,
            payload_json=req.payload_json,
        )
    except ValueError as exc:
        raise _bad_request_from_value_error(exc) from exc
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return _memory_payload(memory)


@router.delete("/memories/{memory_id}")
async def delete_memory_endpoint(
    memory_id: str,
    embedding_model: Optional[str] = Query(default=None),
):
    try:
        result = await hard_delete_memory(memory_id, embedding_model=embedding_model)
    except MemoryVectorCleanupError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if not result["deleted"]:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted", "memory_id": memory_id, "vector_cleanup": result["vector_cleanup"]}


@router.post("/threads/{thread_id}/memories/search")
async def search_thread_memories_endpoint(thread_id: str, req: MemorySearchRequest):
    embedding_model = req.embedding_model
    if not embedding_model:
        from app.db import get_thread

        thread = await get_thread(thread_id)
        if thread is None:
            raise HTTPException(status_code=404, detail="Thread not found")
        embedding_model = thread.embedding_model
    return await search_thread_memory(
        thread_id=thread_id,
        query=req.query,
        embedding_model=embedding_model,
        allowed_scopes=req.allowed_scopes,
        max_results=req.max_results,
    )


@router.get("/memory-candidates")
async def list_memory_candidates_endpoint(
    status: str = Query(default=MemoryCandidateStatus.PENDING.value),
    source_project_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    candidates = await list_memory_candidates(status=status, source_project_id=source_project_id, limit=limit)
    return {"memory_candidates": [_candidate_payload(candidate) for candidate in candidates]}


@router.post("/memory-candidates")
async def create_memory_candidate_endpoint(req: MemoryCandidateCreateRequest):
    try:
        candidate = await create_memory_candidate(
            proposed_scope_type=req.proposed_scope_type,
            proposed_scope_id=req.proposed_scope_id,
            memory_type=req.memory_type,
            content=req.content,
            source_thread_id=req.source_thread_id,
            source_project_id=req.source_project_id,
            source_agent_run_id=req.source_agent_run_id,
            source_turn_id=req.source_turn_id,
            confidence=req.confidence,
            reason=req.reason,
            created_by=req.created_by,
        )
    except ValueError as exc:
        raise _bad_request_from_value_error(exc) from exc
    return _candidate_payload(candidate)


@router.post("/memory-candidates/{candidate_id}/resolve")
async def resolve_memory_candidate_endpoint(candidate_id: str, req: MemoryCandidateResolveRequest):
    repo = get_memory_repo()
    existing = await repo.get_candidate(candidate_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Memory candidate not found")
    if (
        req.status == MemoryCandidateStatus.AUTO_APPROVED.value
        and existing.proposed_scope_type == MemoryScopeType.USER.value
    ):
        raise HTTPException(status_code=400, detail="User/global memory candidates require explicit approval")
    try:
        candidate = await repo.resolve_candidate(candidate_id, status=req.status)
    except ValueError as exc:
        raise _bad_request_from_value_error(exc) from exc
    memory = None
    if req.status in {MemoryCandidateStatus.APPROVED.value, MemoryCandidateStatus.AUTO_APPROVED.value}:
        memory = await create_and_index_memory(
            embedding_model=req.embedding_model,
            scope_type=candidate.proposed_scope_type,
            scope_id=candidate.proposed_scope_id,
            memory_type=candidate.memory_type,
            content=candidate.content,
            summary="",
            source_refs_json={
                "candidate_id": candidate.id,
                "source_thread_id": candidate.source_thread_id,
                "source_project_id": candidate.source_project_id,
                "source_agent_run_id": candidate.source_agent_run_id,
                "source_turn_id": candidate.source_turn_id,
            },
            confidence=candidate.confidence,
            visibility="project" if candidate.proposed_scope_type == MemoryScopeType.PROJECT.value else "private",
            created_by=req.actor_id or candidate.created_by,
            event_payload={"candidate_id": candidate.id},
        )
    return {
        "memory_candidate": _candidate_payload(candidate),
        "memory": _memory_payload(memory) if memory is not None else None,
    }


@router.delete("/memory-candidates/{candidate_id}")
async def delete_memory_candidate_endpoint(candidate_id: str):
    result = await hard_delete_memory_candidate(candidate_id)
    if not result["deleted"]:
        raise HTTPException(status_code=404, detail="Memory candidate not found")
    return {"status": "deleted", "memory_candidate_id": candidate_id}
