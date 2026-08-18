"""Run-scoped background task ownership for cancellable tool work."""

import asyncio
from collections import defaultdict

_tasks: dict[str, set[asyncio.Task]] = defaultdict(set)


def register_background_task(scope_id: str | None, task: asyncio.Task) -> asyncio.Task:
    if not scope_id:
        return task
    bucket = _tasks[scope_id]
    bucket.add(task)
    task.add_done_callback(lambda completed: _discard(scope_id, completed))
    return task


def _discard(scope_id: str, task: asyncio.Task) -> None:
    bucket = _tasks.get(scope_id)
    if bucket is None:
        return
    bucket.discard(task)
    if not bucket:
        _tasks.pop(scope_id, None)


async def cancel_background_tasks(scope_id: str | None) -> None:
    tasks = list(_tasks.pop(scope_id, set())) if scope_id else []
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
