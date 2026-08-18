"""Embedding model ownership, scope resolution, and readiness enforcement."""

from __future__ import annotations

from dataclasses import dataclass

from app.db import MemoryScopeType, get_project, get_thread
from app.models.llm_server_client import (
    LOCAL_EMBEDDING_MODEL,
    check_embedding_model_ready,
)


GLOBAL_MEMORY_EMBEDDING_MODEL = LOCAL_EMBEDDING_MODEL


class EmbeddingModelResolutionError(ValueError):
    """Raised when an embedding model cannot be resolved from app-owned scope."""


class EmbeddingModelUnavailableError(RuntimeError):
    """Raised when an embedding-dependent operation targets an unavailable model."""


@dataclass(frozen=True)
class ThreadEmbeddingContext:
    thread: object
    project: object
    embedding_model: str


async def resolve_thread_embedding_context(thread_id: str) -> ThreadEmbeddingContext:
    thread = await get_thread(thread_id)
    if thread is None:
        raise EmbeddingModelResolutionError("Thread not found")
    project = await get_project(thread.project_id)
    if project is None:
        raise EmbeddingModelResolutionError("Thread project not found")
    if thread.embedding_model != project.embedding_model:
        raise EmbeddingModelResolutionError(
            "Thread embedding model is inconsistent with its project"
        )
    return ThreadEmbeddingContext(
        thread=thread,
        project=project,
        embedding_model=project.embedding_model,
    )


async def resolve_scope_embedding_model(scope_type: str, scope_id: str) -> str:
    if scope_type == MemoryScopeType.USER.value:
        return GLOBAL_MEMORY_EMBEDDING_MODEL
    if scope_type == MemoryScopeType.PROJECT.value:
        project = await get_project(scope_id)
        if project is None:
            raise EmbeddingModelResolutionError("Project memory scope not found")
        return project.embedding_model
    if scope_type == MemoryScopeType.THREAD.value:
        context = await resolve_thread_embedding_context(scope_id)
        return context.embedding_model
    raise EmbeddingModelResolutionError(f"Unsupported memory scope_type: {scope_type}")


async def require_embedding_model_ready(model: str) -> None:
    if not await check_embedding_model_ready(model):
        raise EmbeddingModelUnavailableError(
            f"Embedding model '{model}' is unavailable"
        )


async def require_thread_embedding_ready(thread_id: str) -> ThreadEmbeddingContext:
    context = await resolve_thread_embedding_context(thread_id)
    await require_embedding_model_ready(context.embedding_model)
    return context
