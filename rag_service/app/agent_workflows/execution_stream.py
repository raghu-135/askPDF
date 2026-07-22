from __future__ import annotations

import asyncio
from typing import Any, Dict

from app.agent_workflows.trace_sanitization import _bounded_value


_background_tasks: set[asyncio.Task[Any]] = set()


class AgentExecutionEventSink:
    """Per-request event subscriber used by persisted chat and resume streams."""

    def __init__(self, *, include_details: bool = False):
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.include_details = include_details
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def _event(self, event: str, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = dict(data or {})
        if not self.include_details:
            payload.pop("detail", None)
            payload.pop("checkpoint_before", None)
            payload.pop("checkpoint_after", None)
            payload.pop("prompt", None)
            payload.pop("reasoning", None)
            payload.pop("tools", None)
        return {"event": event, "data": _bounded_value(payload)}

    async def emit(self, event: str, data: Dict[str, Any] | None = None) -> None:
        if not self.closed:
            await self.queue.put(self._event(event, data))

    def emit_nowait(self, event: str, data: Dict[str, Any] | None = None) -> None:
        if not self.closed:
            self.queue.put_nowait(self._event(event, data))


def retain_background_task(task: asyncio.Task[Any]) -> None:
    """Keep disconnected chat executions alive until persistence finishes."""

    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
