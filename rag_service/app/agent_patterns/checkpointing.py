from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Sequence

from langgraph.checkpoint.memory import InMemorySaver


logger = logging.getLogger(__name__)

_MEMORY_CHECKPOINTER = InMemorySaver()


def _truthy_env(name: str, default: str = "") -> bool:
    value = os.environ.get(name, default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def final_review_hitl_enabled() -> bool:
    """Return whether the runtime should inject the first checkpointed HITL gate."""

    return _truthy_env("ASKPDF_AGENT_HITL_FINAL_REVIEW")


def _postgres_checkpoint_url() -> str:
    url = os.environ.get("AGENT_CHECKPOINT_DATABASE_URL") or os.environ.get("DATABASE_URL") or ""
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + url[len("postgresql+asyncpg://"):]
    return url


def _memory_fallback_allowed() -> bool:
    return _truthy_env("ASKPDF_AGENT_CHECKPOINTER_ALLOW_MEMORY_FALLBACK")


@asynccontextmanager
async def open_agent_checkpointer() -> AsyncIterator[Any]:
    """Yield a LangGraph checkpointer.

    Production can opt into LangGraph's Postgres saver with
    ASKPDF_AGENT_CHECKPOINTER=postgres. Tests and local dev use one process-wide
    in-memory saver so a run can pause in one service call and resume in another.
    An explicit Postgres mode fails closed unless
    ASKPDF_AGENT_CHECKPOINTER_ALLOW_MEMORY_FALLBACK=true is set.
    """

    mode = os.environ.get("ASKPDF_AGENT_CHECKPOINTER", "memory").strip().lower()
    if mode == "memory":
        yield _MEMORY_CHECKPOINTER
        return
    if mode != "postgres":
        raise ValueError(f"Unsupported ASKPDF_AGENT_CHECKPOINTER value: {mode!r}")

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except Exception as exc:
        if _memory_fallback_allowed():
            logger.warning(
                "LangGraph Postgres checkpointer unavailable; falling back to in-memory checkpointer: %s",
                exc,
            )
            yield _MEMORY_CHECKPOINTER
            return
        raise RuntimeError("LangGraph Postgres checkpointer is required but unavailable") from exc

    checkpoint_url = _postgres_checkpoint_url()
    if not checkpoint_url:
        if _memory_fallback_allowed():
            logger.warning("No Postgres URL configured for agent checkpointer; falling back to in-memory checkpointer")
            yield _MEMORY_CHECKPOINTER
            return
        raise RuntimeError("ASKPDF_AGENT_CHECKPOINTER=postgres requires AGENT_CHECKPOINT_DATABASE_URL or DATABASE_URL")

    async with AsyncPostgresSaver.from_conn_string(checkpoint_url) as checkpointer:
        if _truthy_env("ASKPDF_AGENT_CHECKPOINTER_SETUP", "true"):
            await checkpointer.setup()
        yield checkpointer


async def delete_agent_checkpoints(
    checkpoint_thread_ids: Sequence[str],
    *,
    checkpointer: Any = None,
) -> list[str]:
    """Delete LangGraph checkpoints for completed/abandoned askPDF run threads."""

    unique_ids = []
    seen = set()
    for checkpoint_thread_id in checkpoint_thread_ids:
        value = str(checkpoint_thread_id or "").strip()
        if value and value not in seen:
            seen.add(value)
            unique_ids.append(value)
    if not unique_ids:
        return []

    async def _delete_with(active_checkpointer: Any) -> list[str]:
        deleted: list[str] = []
        delete_thread = getattr(active_checkpointer, "adelete_thread", None)
        if delete_thread is None:
            raise RuntimeError("Configured LangGraph checkpointer does not support thread deletion")
        for checkpoint_thread_id in unique_ids:
            await delete_thread(checkpoint_thread_id)
            deleted.append(checkpoint_thread_id)
        return deleted

    if checkpointer is not None:
        return await _delete_with(checkpointer)

    async with open_agent_checkpointer() as active_checkpointer:
        return await _delete_with(active_checkpointer)
