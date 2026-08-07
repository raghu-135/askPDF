from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

from app.agent_workflows.trace_sanitization import _bounded_value
from app.agent_workflows.parallel_contracts import PARALLEL_EVENT_JOURNAL_LIMIT, PARALLEL_EVENT_PREFIXES
from app.agent_workflows.parallel_observability import enrich_parallel_event


_background_tasks: set[asyncio.Task[Any]] = set()


class AgentExecutionEventSink:
    """Per-request event subscriber used by persisted chat and resume streams."""

    def __init__(self, *, include_details: bool = False):
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.include_details = include_details
        self.closed = False
        self._emit_lock = asyncio.Lock()
        self._parallel_events: list[Dict[str, Any]] = []
        self._runtime_event_ids: set[str] = set()
        self._trace_recorder: Any = None

    def bind_trace_recorder(self, recorder: Any) -> None:
        self._trace_recorder = recorder

    def parallel_events(self) -> list[Dict[str, Any]]:
        return [dict(item) for item in self._parallel_events]

    def close(self) -> None:
        self.closed = True

    def _event(self, event: str, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = enrich_parallel_event(event, data or {}) if event.startswith(PARALLEL_EVENT_PREFIXES) else dict(data or {})
        if event.startswith(PARALLEL_EVENT_PREFIXES):
            payload.setdefault("occurred_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        if not self.include_details:
            payload.pop("detail", None)
            payload.pop("checkpoint_before", None)
            payload.pop("checkpoint_after", None)
            payload.pop("prompt", None)
            payload.pop("reasoning", None)
            payload.pop("tools", None)
        return {"event": event, "data": _bounded_value(payload)}

    async def emit(self, event: str, data: Dict[str, Any] | None = None) -> None:
        if self.closed:
            return
        envelope = self._event(event, data)
        async with self._emit_lock:
            event_id = str((envelope.get("data") or {}).get("event_id") or "")
            if event_id and event_id in self._runtime_event_ids:
                return
            if event_id:
                self._runtime_event_ids.add(event_id)
            if event.startswith(PARALLEL_EVENT_PREFIXES):
                self._parallel_events.append(envelope)
                if len(self._parallel_events) > PARALLEL_EVENT_JOURNAL_LIMIT:
                    del self._parallel_events[:-PARALLEL_EVENT_JOURNAL_LIMIT]
                if self._trace_recorder is not None and hasattr(self._trace_recorder, "record_runtime_event"):
                    self._trace_recorder.record_runtime_event(
                        event,
                        attributes=envelope.get("data") or {},
                    )
            await self.queue.put(envelope)

    def emit_nowait(self, event: str, data: Dict[str, Any] | None = None) -> None:
        if not self.closed:
            envelope = self._event(event, data)
            event_id = str((envelope.get("data") or {}).get("event_id") or "")
            if event_id and event_id in self._runtime_event_ids:
                return
            if event_id:
                self._runtime_event_ids.add(event_id)
            self.queue.put_nowait(envelope)


def retain_background_task(task: asyncio.Task[Any]) -> None:
    """Keep disconnected chat executions alive until persistence finishes."""

    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
