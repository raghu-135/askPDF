"""Owned, deduplicated in-process scheduling for deterministic memory repairs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any


logger = logging.getLogger(__name__)
_repair_tasks: dict[str, asyncio.Task[Any]] = {}


def schedule_memory_repair(
    key: str,
    repair: Callable[[], Awaitable[Any]],
) -> asyncio.Task[Any]:
    """Schedule one repair per key and consume every terminal exception."""

    normalized_key = str(key or "").strip()
    if not normalized_key:
        raise ValueError("memory repair key is required")
    existing = _repair_tasks.get(normalized_key)
    if existing is not None and not existing.done():
        return existing

    task = asyncio.create_task(repair(), name=f"memory-repair:{normalized_key}")
    _repair_tasks[normalized_key] = task

    def consume_result(completed: asyncio.Task[Any]) -> None:
        if _repair_tasks.get(normalized_key) is completed:
            _repair_tasks.pop(normalized_key, None)
        try:
            completed.result()
        except asyncio.CancelledError:
            logger.info("Memory repair cancelled | key=%s", normalized_key)
        except Exception:
            logger.exception("Memory repair failed | key=%s", normalized_key)

    task.add_done_callback(consume_result)
    return task


def schedule_global_representation_repair(embedding_model: str) -> asyncio.Task[Any]:
    """Deduplicate global representation warming by target embedding model."""

    model = str(embedding_model or "").strip()
    if not model:
        raise ValueError("embedding_model is required")

    async def repair() -> Any:
        from app.services.memory_representation_service import warm_global_representations_for_model
        return await warm_global_representations_for_model(model)

    return schedule_memory_repair(f"global-representations:{model}", repair)


def pending_memory_repair_keys() -> tuple[str, ...]:
    return tuple(sorted(key for key, task in _repair_tasks.items() if not task.done()))


async def shutdown_memory_repairs() -> None:
    """Cancel and consume owned repair tasks before service resources close."""

    tasks = [task for task in _repair_tasks.values() if not task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    _repair_tasks.clear()

