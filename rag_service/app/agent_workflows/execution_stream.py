from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

from app.agent_workflows.trace_sanitization import _bounded_value
from app.agent_workflows.parallel_contracts import PARALLEL_EVENT_JOURNAL_LIMIT, PARALLEL_EVENT_PREFIXES
from app.agent_workflows.parallel_observability import enrich_parallel_event
from app.runtime.contracts import AgentRuntimeEvent
from app.runtime.events import create_runtime_event, validate_runtime_event
from app.runtime.observability import normalize_runtime_event, project_event_to_trace_recorder


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
        self._runtime_binding_persister: Any = None
        self._runtime_event_persister: Any = None
        self._run_id: str | None = None
        self._sequence = 0
        self._canonical_events: list[AgentRuntimeEvent] = []

    def bind_trace_recorder(self, recorder: Any) -> None:
        self._trace_recorder = recorder

    def bind_runtime_binding_persister(self, persister: Any) -> None:
        self._runtime_binding_persister = persister

    def bind_runtime_event_persister(self, run_id: str, persister: Any) -> None:
        self._run_id = run_id
        self._runtime_event_persister = persister

    def canonical_events(self) -> list[AgentRuntimeEvent]:
        return list(self._canonical_events)

    async def persist_runtime_binding(self, run_id: str, binding: Any) -> None:
        if self._runtime_binding_persister is not None:
            await self._runtime_binding_persister(run_id, binding)

    def parallel_events(self) -> list[Dict[str, Any]]:
        return [dict(item) for item in self._parallel_events]

    def close(self) -> None:
        self.closed = True

    def _event(self, event: str, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = enrich_parallel_event(event, data or {}) if event.startswith(PARALLEL_EVENT_PREFIXES) else dict(data or {})
        public_event, _ = normalize_runtime_event(event, payload)
        if event.startswith(PARALLEL_EVENT_PREFIXES):
            payload.setdefault("occurred_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        if not self.include_details:
            payload.pop("detail", None)
            payload.pop("checkpoint_before", None)
            payload.pop("checkpoint_after", None)
            payload.pop("prompt", None)
            payload.pop("reasoning", None)
            payload.pop("tools", None)
        return {"event": public_event, "data": _bounded_value(payload)}

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
            self._sequence += 1
            normalized_kind, normalized_payload = normalize_runtime_event(event, envelope.get("data") or {})
            source_metadata = dict(normalized_payload.get("source_metadata") or {})
            source_sequence = normalized_payload.pop("sequence", None)
            if source_sequence is not None:
                source_metadata.setdefault("source_sequence", source_sequence)
            if normalized_kind != event:
                source_metadata.setdefault("source_event", event)
            canonical = create_runtime_event(
                event_id=event_id or f"{self._run_id or 'run'}:{self._sequence}",
                run_id=self._run_id or str(normalized_payload.get("run_id") or ""),
                sequence=self._sequence,
                attempt=int(normalized_payload.get("attempt") or 1),
                kind=normalized_kind,
                payload=normalized_payload,
                occurred_at=normalized_payload.get("occurred_at") or normalized_payload.get("timestamp"),
                trace_id=normalized_payload.get("trace_id"),
                source_metadata=source_metadata,
            )
            validate_runtime_event(canonical, previous=self._canonical_events[-1] if self._canonical_events else None)
            self._canonical_events.append(canonical)
            if self._runtime_event_persister is not None and canonical.run_id:
                await self._runtime_event_persister(canonical.run_id, canonical)
            if self._trace_recorder is not None:
                project_event_to_trace_recorder(self._trace_recorder, canonical.kind, canonical.payload)
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

    async def emit_runtime_event(self, event: AgentRuntimeEvent) -> None:
        payload = dict(event.payload or {})
        payload.setdefault("event_id", event.event_id)
        payload.setdefault("sequence", event.sequence)
        payload.setdefault("attempt", event.attempt)
        if event.occurred_at:
            payload.setdefault("occurred_at", event.occurred_at)
        if event.trace_id:
            payload.setdefault("trace_id", event.trace_id)
        if event.source_metadata:
            payload.setdefault("source_metadata", dict(event.source_metadata))
        await self.emit(event.kind, payload)

    def emit_nowait(self, event: str, data: Dict[str, Any] | None = None) -> None:
        if not self.closed:
            envelope = self._event(event, data)
            event_id = str((envelope.get("data") or {}).get("event_id") or "")
            if event_id and event_id in self._runtime_event_ids:
                return
            if event_id:
                self._runtime_event_ids.add(event_id)
            self._sequence += 1
            normalized_kind, normalized_payload = normalize_runtime_event(event, envelope.get("data") or {})
            source_metadata = dict(normalized_payload.get("source_metadata") or {})
            source_sequence = normalized_payload.pop("sequence", None)
            if source_sequence is not None:
                source_metadata.setdefault("source_sequence", source_sequence)
            if normalized_kind != event:
                source_metadata.setdefault("source_event", event)
            canonical = create_runtime_event(
                event_id=event_id or f"{self._run_id or 'run'}:{self._sequence}",
                run_id=self._run_id or str(normalized_payload.get("run_id") or ""),
                sequence=self._sequence,
                attempt=int(normalized_payload.get("attempt") or 1),
                kind=normalized_kind,
                payload=normalized_payload,
                occurred_at=normalized_payload.get("occurred_at") or normalized_payload.get("timestamp"),
                trace_id=normalized_payload.get("trace_id"),
                source_metadata=source_metadata,
            )
            validate_runtime_event(canonical, previous=self._canonical_events[-1] if self._canonical_events else None)
            self._canonical_events.append(canonical)
            if self._trace_recorder is not None:
                project_event_to_trace_recorder(self._trace_recorder, canonical.kind, canonical.payload)
            if self._runtime_event_persister is not None and canonical.run_id:
                retain_background_task(asyncio.create_task(self._runtime_event_persister(canonical.run_id, canonical)))
            self.queue.put_nowait(envelope)


def retain_background_task(task: asyncio.Task[Any]) -> None:
    """Keep disconnected chat executions alive until persistence finishes."""

    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
