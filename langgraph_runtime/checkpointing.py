from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Sequence

from langgraph.checkpoint.memory import InMemorySaver

from langgraph_runtime.workflows.enums import AgentCheckpointerMode
from langgraph_runtime.capabilities import checkpoint_database_url


logger = logging.getLogger(__name__)

_MEMORY_CHECKPOINTER = InMemorySaver()
_POSTGRES_SETUP_LOCK = asyncio.Lock()
_POSTGRES_SETUP_COMPLETE: set[str] = set()


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _postgres_checkpoint_url() -> str:
    return checkpoint_database_url(os.environ)


@asynccontextmanager
async def open_agent_checkpointer(*, setup: bool = True) -> AsyncIterator[Any]:
    """Yield a LangGraph checkpointer.

    Production can opt into LangGraph's Postgres saver with
    ASKPDF_AGENT_CHECKPOINTER=postgres. Tests and local dev use one process-wide
    in-memory saver so a run can pause in one service call and resume in another.
    An explicit Postgres mode fails closed when the configured saver is unavailable.
    """

    mode = os.environ.get("ASKPDF_AGENT_CHECKPOINTER", "").strip().lower()
    if mode == AgentCheckpointerMode.MEMORY.value:
        yield _MEMORY_CHECKPOINTER
        return
    if mode != AgentCheckpointerMode.POSTGRES.value:
        raise ValueError(f"Unsupported ASKPDF_AGENT_CHECKPOINTER value: {mode!r}")

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except Exception as exc:
        raise RuntimeError("LangGraph Postgres checkpointer is required but unavailable") from exc

    checkpoint_url = _postgres_checkpoint_url()
    if not checkpoint_url:
        raise RuntimeError("ASKPDF_AGENT_CHECKPOINTER=postgres requires AGENT_CHECKPOINT_DATABASE_URL or DATABASE_URL")

    async with AsyncPostgresSaver.from_conn_string(checkpoint_url) as checkpointer:
        if setup and _truthy_env("ASKPDF_AGENT_CHECKPOINTER_SETUP"):
            if checkpoint_url not in _POSTGRES_SETUP_COMPLETE:
                async with _POSTGRES_SETUP_LOCK:
                    if checkpoint_url not in _POSTGRES_SETUP_COMPLETE:
                        await checkpointer.setup()
                        _POSTGRES_SETUP_COMPLETE.add(checkpoint_url)
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
