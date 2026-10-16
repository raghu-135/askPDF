"""Framework-neutral cooperative cancellation primitives."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from app.runtime.operational_limits import required_positive_float


T = TypeVar("T")
CancellationChecker = Callable[[], Awaitable[bool]]


async def race_with_cancellation(
    awaitable: Awaitable[T],
    checker: CancellationChecker,
    *,
    timeout_seconds: float | None = None,
    poll_seconds: float | None = None,
) -> T:
    """Race work against durable cancellation and always gather both tasks."""

    interval = poll_seconds or required_positive_float("AGENT_CANCELLATION_POLL_INTERVAL_SECONDS")
    work = asyncio.create_task(awaitable)

    async def observe() -> None:
        while not await checker():
            await asyncio.sleep(interval)

    cancellation = asyncio.create_task(observe())
    try:
        done, _ = await asyncio.wait(
            {work, cancellation}, timeout=timeout_seconds, return_when=asyncio.FIRST_COMPLETED,
        )
        if work in done:
            return await work
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        if cancellation in done:
            await cancellation
            raise asyncio.CancelledError
        raise asyncio.TimeoutError
    finally:
        for task in (work, cancellation):
            if not task.done():
                task.cancel()
        await asyncio.gather(work, cancellation, return_exceptions=True)
