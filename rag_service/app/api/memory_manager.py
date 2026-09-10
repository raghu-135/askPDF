"""Unified memory planning and application API."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.models.requests import (
    MemoryManagerApplyRequest,
    MemoryManagerContext,
    MemoryManagerPlanRequest,
)
from app.services.embedding_model_service import EmbeddingModelUnavailableError
from app.services.memory_manager_engine import MemoryChangedError
from app.services.memory_manager_service import (
    apply_memory_manager_plan,
    create_memory_manager_plan,
)


router = APIRouter(prefix="/memory-manager", tags=["memory-manager"])
logger = logging.getLogger(__name__)


def _raise_memory_manager_http(
    exc: Exception,
    *,
    context: MemoryManagerContext | None = None,
    operation: str,
) -> None:
    logger.exception(
        "Memory-manager request failed",
        extra={
            "error_code": getattr(exc, "code", "memory_manager_error"),
            "operation": operation,
            "thread_id": context.thread_id if context else None,
            "project_id": context.project_id if context else None,
            "memory_scope_type": context.selected_scope_type if context else None,
            "memory_scope_id": context.selected_scope_id if context else None,
        },
    )
    if isinstance(exc, (MemoryChangedError, EmbeddingModelUnavailableError)):
        raise HTTPException(
            status_code=409,
            detail={"code": getattr(exc, "code", "memory_conflict"), "message": str(exc)},
        ) from exc
    raise HTTPException(
        status_code=400,
        detail={"code": getattr(exc, "code", "memory_manager_error"), "message": str(exc)},
    ) from exc


@router.post("/plan")
async def memory_manager_plan_endpoint(req: MemoryManagerPlanRequest):
    try:
        return await create_memory_manager_plan(req)
    except Exception as exc:
        _raise_memory_manager_http(exc, context=req.context, operation="plan")


@router.post("/apply")
async def memory_manager_apply_endpoint(req: MemoryManagerApplyRequest):
    try:
        return await apply_memory_manager_plan(req)
    except Exception as exc:
        _raise_memory_manager_http(exc, context=req.plan.context, operation="apply")


@router.post("/reviews")
async def memory_manager_review_start_endpoint(req: MemoryManagerPlanRequest):
    if req.mode != "consistency_review":
        raise HTTPException(status_code=422, detail="Review requests must use consistency_review mode")
    try:
        return await create_memory_manager_plan(req)
    except Exception as exc:
        _raise_memory_manager_http(exc, context=req.context, operation="review_start")


@router.post("/reviews/{review_id}/continue")
async def memory_manager_review_continue_endpoint(review_id: str, req: MemoryManagerPlanRequest):
    if req.mode != "consistency_review":
        raise HTTPException(status_code=422, detail="Review requests must use consistency_review mode")
    try:
        return await create_memory_manager_plan(req.model_copy(update={"review_id": review_id}))
    except Exception as exc:
        _raise_memory_manager_http(exc, context=req.context, operation="review_continue")


@router.get("/reviews/{review_id}")
async def memory_manager_review_status_endpoint(review_id: str):
    return {
        "review_id": review_id,
        "state": "client_held",
        "message": "Review state is held by the client; submit the saved cursor to continue.",
    }
