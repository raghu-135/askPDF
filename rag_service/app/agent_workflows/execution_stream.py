from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict

from app.agent_workflows.canonical_trace import build_parallel_groups
from app.agent_workflows.parallel_projection_contracts import PARALLEL_EVENT_JOURNAL_LIMIT, PARALLEL_EVENT_PREFIXES
from app.agent_workflows.parallel_observability import enrich_parallel_event
from app.agent_workflows.trace_sanitization import _bounded_value
from runtime_protocol.contracts import AgentRuntimeEvent, ContinuationBinding
from runtime_protocol.events import RuntimeEventContractViolation, create_runtime_event, validate_runtime_event
from app.runtime.observability import normalize_runtime_event


logger = logging.getLogger(__name__)


@dataclass
class _EventCommand:
    event: str
    data: Dict[str, Any]
    acknowledgement: asyncio.Future[None] | None = None
    terminal: bool = False
    terminal_committer: Any = None


_retained_executions: set[asyncio.Task[Any]] = set()


class AgentExecutionEventSink:
    """Ordered per-run event persistence with an optional live subscriber."""

    def __init__(self, *, include_details: bool = False):
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self.include_details = include_details
        self._commands: asyncio.Queue[_EventCommand | None] = asyncio.Queue()
        self._writer_task: asyncio.Task[None] | None = None
        self._delivery_attached = True
        self._accepting = True
        self._finished = False
        self._writer_error: BaseException | None = None
        self._parallel_events: list[Dict[str, Any]] = []
        self._runtime_event_ids: dict[str, str] = {}
        self._trace_recorder: Any = None
        self._runtime_binding_persister: Any = None
        self._runtime_fact_persister: Any = None
        self._runtime_event_persister: Any = None
        self._run_id: str | None = None
        self._sequence = 0
        self._canonical_events: list[AgentRuntimeEvent] = []

    def bind_trace_recorder(self, recorder: Any) -> None:
        self._trace_recorder = recorder

    def bind_runtime_binding_persister(self, persister: Any) -> None:
        self._runtime_binding_persister = persister

    def bind_runtime_fact_persister(self, persister: Any) -> None:
        self._runtime_fact_persister = persister

    def bind_runtime_event_persister(
        self,
        run_id: str,
        persister: Any,
        *,
        initial_sequence: int = 0,
    ) -> None:
        if self._writer_task is not None or not self._commands.empty():
            raise RuntimeError("Runtime event persistence must be bound before events are emitted")
        self._run_id = run_id
        self._runtime_event_persister = persister
        self._sequence = max(0, int(initial_sequence))

    def canonical_events(self) -> list[AgentRuntimeEvent]:
        return list(self._canonical_events)

    async def persist_runtime_binding(self, run_id: str, binding: Any) -> None:
        if self._runtime_binding_persister is not None:
            await self._runtime_binding_persister(run_id, binding)

    def parallel_events(self) -> list[Dict[str, Any]]:
        return [dict(item) for item in self._parallel_events]

    def detach_delivery(self) -> None:
        self._delivery_attached = False

    def _ensure_writer(self) -> None:
        if self._writer_task is None:
            self._writer_task = asyncio.create_task(
                self._writer(), name=f"agent-event-writer-{self._run_id or 'unbound'}"
            )

    def _event(self, event: str, data: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload = enrich_parallel_event(event, data or {}) if event.startswith(PARALLEL_EVENT_PREFIXES) else dict(data or {})
        public_event, _ = normalize_runtime_event(event, payload)
        if event.startswith(PARALLEL_EVENT_PREFIXES):
            payload.setdefault("occurred_at", datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        if not self.include_details:
            for key in ("detail", "checkpoint_before", "checkpoint_after", "prompt", "reasoning", "tools"):
                payload.pop(key, None)
        return {"event": public_event, "data": _bounded_value(payload)}

    def _enqueue(
        self,
        event: str,
        data: Dict[str, Any] | None,
        *,
        acknowledgement: asyncio.Future[None] | None,
        terminal: bool = False,
    ) -> None:
        if not self._accepting and not terminal:
            logger.warning("Ignoring runtime event after sink finalization | run_id=%s event=%s", self._run_id, event)
            if acknowledgement is not None and not acknowledgement.done():
                acknowledgement.set_exception(RuntimeError("Runtime event sink is finalized"))
            return
        self._ensure_writer()
        self._commands.put_nowait(_EventCommand(event, dict(data or {}), acknowledgement, terminal))

    async def emit(self, event: str, data: Dict[str, Any] | None = None) -> None:
        if self._writer_error is not None:
            raise self._writer_error
        acknowledgement = asyncio.get_running_loop().create_future()
        self._enqueue(event, data, acknowledgement=acknowledgement)
        await acknowledgement

    async def emit_runtime_event(self, event: AgentRuntimeEvent) -> None:
        # A transport terminal closes a runtime operation. The product service
        # publishes its terminal only after projection and persistence commit.
        if event.terminal:
            return
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
        if event.continuation is not None:
            payload["_runtime_continuation"] = event.continuation.to_dict()
        if event.checkpoint_boundary_available is not None:
            payload["_checkpoint_boundary_available"] = bool(event.checkpoint_boundary_available)
        await self.emit(event.kind, payload)

    def emit_nowait(self, event: str, data: Dict[str, Any] | None = None) -> None:
        if self._writer_error is not None:
            logger.error("Dropping runtime event after writer failure | run_id=%s event=%s", self._run_id, event)
            return
        self._enqueue(event, data, acknowledgement=None)

    async def flush(self) -> None:
        if self._writer_error is not None:
            raise self._writer_error
        if self._writer_task is None:
            return
        acknowledgement = asyncio.get_running_loop().create_future()
        self._commands.put_nowait(_EventCommand("__flush__", {}, acknowledgement))
        await acknowledgement
        if self._writer_error is not None:
            raise self._writer_error

    async def finish(
        self,
        event: str,
        data: Dict[str, Any] | None = None,
        *,
        terminal_committer: Any = None,
    ) -> None:
        if self._finished:
            return
        self._accepting = False
        acknowledgement = asyncio.get_running_loop().create_future()
        self._ensure_writer()
        self._commands.put_nowait(
            _EventCommand(event, dict(data or {}), acknowledgement, True, terminal_committer)
        )
        try:
            await acknowledgement
        finally:
            await self._stop_writer()
            self._finished = True

    async def finish_boundary(self) -> None:
        if self._finished:
            return
        self._accepting = False
        try:
            await self.flush()
        finally:
            await self._stop_writer()
            self._finished = True

    async def _stop_writer(self) -> None:
        if self._writer_task is None:
            return
        self._commands.put_nowait(None)
        await self._writer_task
        self._writer_task = None

    async def _writer(self) -> None:
        while True:
            command = await self._commands.get()
            if command is None:
                return
            if command.event == "__flush__":
                if command.acknowledgement is not None and not command.acknowledgement.done():
                    command.acknowledgement.set_result(None)
                continue
            if self._writer_error is not None and not command.terminal:
                if command.acknowledgement is not None and not command.acknowledgement.done():
                    command.acknowledgement.set_exception(self._writer_error)
                continue
            try:
                await self._record_event(
                    command.event,
                    command.data,
                    product_terminal=command.terminal,
                    terminal_committer=command.terminal_committer,
                )
            except Exception as exc:
                self._writer_error = exc
                logger.exception("Runtime event writer failed | run_id=%s event=%s", self._run_id, command.event)
                if command.acknowledgement is not None and not command.acknowledgement.done():
                    command.acknowledgement.set_exception(exc)
                continue
            if command.acknowledgement is not None and not command.acknowledgement.done():
                command.acknowledgement.set_result(None)

    async def _record_event(
        self,
        event: str,
        data: Dict[str, Any],
        *,
        product_terminal: bool = False,
        terminal_committer: Any = None,
    ) -> None:
        envelope = self._event(event, data)
        event_id = str((envelope.get("data") or {}).get("event_id") or "")
        normalized_kind, normalized_payload = normalize_runtime_event(event, envelope.get("data") or {})
        source_metadata = dict(normalized_payload.get("source_metadata") or {})
        continuation_value = normalized_payload.pop("_runtime_continuation", None)
        continuation = (
            ContinuationBinding(
                binding_type=str(continuation_value.get("binding_type") or ""),
                payload=dict(continuation_value.get("payload") or {}),
            )
            if isinstance(continuation_value, dict) and continuation_value.get("binding_type")
            else None
        )
        checkpoint_boundary_available = normalized_payload.pop(
            "_checkpoint_boundary_available", None
        )
        source_sequence = normalized_payload.pop("sequence", None)
        if source_sequence is not None:
            source_metadata.setdefault("source_sequence", source_sequence)
        if normalized_kind != event:
            source_metadata.setdefault("source_event", event)
        hash_payload = {
            key: value for key, value in normalized_payload.items()
            if key not in {"occurred_at", "timestamp"}
        }
        candidate_hash = hashlib.sha256(
            json.dumps(
                {"kind": normalized_kind, "payload": hash_payload, "source_metadata": source_metadata},
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        if event_id and event_id in self._runtime_event_ids:
            if self._runtime_event_ids[event_id] != candidate_hash:
                raise ValueError(f"Conflicting duplicate runtime event ID: {event_id}")
            return

        self._sequence += 1
        generated_event_id = (
            f"askpdf-terminal:{self._run_id or 'run'}:{normalized_kind}"
            if product_terminal
            else f"{self._run_id or 'run'}:{self._sequence}"
        )
        try:
            canonical = create_runtime_event(
                event_id=event_id or generated_event_id,
                run_id=self._run_id or str(normalized_payload.get("run_id") or normalized_payload.get("agent_run_id") or "unbound"),
                sequence=self._sequence,
                attempt=int(normalized_payload.get("attempt") or 1),
                kind=normalized_kind,
                payload=normalized_payload,
                occurred_at=normalized_payload.get("occurred_at") or normalized_payload.get("timestamp"),
                trace_id=normalized_payload.get("trace_id"),
                source_metadata=source_metadata,
                continuation=continuation,
                checkpoint_boundary_available=checkpoint_boundary_available,
            )
            validate_runtime_event(canonical, previous=self._canonical_events[-1] if self._canonical_events else None)
        except (TypeError, ValueError) as exc:
            raise RuntimeEventContractViolation(
                str(exc),
                correlation_id=f"trace:{self._run_id or 'unbound'}",
            ) from exc
        if terminal_committer is not None:
            # The terminal debug payload is built by the committer, so the
            # canonical terminal event must already be part of the recorder.
            self._canonical_events.append(canonical)
            if event_id:
                self._runtime_event_ids[event_id] = candidate_hash
            if self._trace_recorder is not None:
                self._trace_recorder.record_agent_runtime_event(canonical)
            await terminal_committer(canonical)
        else:
            if self._runtime_event_persister is not None and canonical.run_id:
                await self._runtime_event_persister(canonical.run_id, canonical)
            self._canonical_events.append(canonical)
            if event_id:
                self._runtime_event_ids[event_id] = candidate_hash
            if self._trace_recorder is not None:
                self._trace_recorder.record_agent_runtime_event(canonical)
        if canonical.continuation is not None and self._runtime_binding_persister is not None:
            await self._runtime_binding_persister(canonical.run_id, canonical.continuation)
        if (
            canonical.checkpoint_boundary_available is not None
            and self._runtime_fact_persister is not None
        ):
            await self._runtime_fact_persister(
                canonical.run_id,
                {"checkpoint_boundary_available": canonical.checkpoint_boundary_available},
            )
        if event.startswith(PARALLEL_EVENT_PREFIXES):
            self._parallel_events.append(envelope)
            if len(self._parallel_events) > PARALLEL_EVENT_JOURNAL_LIMIT:
                del self._parallel_events[:-PARALLEL_EVENT_JOURNAL_LIMIT]
            if self._trace_recorder is not None and hasattr(self._trace_recorder, "record_runtime_event"):
                self._trace_recorder.record_runtime_event(event, attributes=envelope.get("data") or {})
        if self._delivery_attached:
            delivery_payload = dict(canonical.payload)
            delivery_payload.setdefault("event_id", canonical.event_id)
            parallel_groups = build_parallel_groups(self._canonical_events)
            if parallel_groups:
                delivery_payload["parallel_groups"] = parallel_groups
            await self.queue.put({"event": canonical.kind, "data": delivery_payload})


def retain_background_task(task: asyncio.Task[Any]) -> None:
    """Retain a disconnected execution and observe its terminal exception."""

    _retained_executions.add(task)

    def completed(value: asyncio.Task[Any]) -> None:
        _retained_executions.discard(value)
        if value.cancelled():
            return
        try:
            error = value.exception()
        except asyncio.CancelledError:
            return
        if error is not None:
            logger.error("Retained agent execution failed", exc_info=(type(error), error, error.__traceback__))

    task.add_done_callback(completed)


async def drain_retained_executions(timeout_seconds: float) -> None:
    """Drain retained executions, then cancel and gather overdue work."""

    pending = set(_retained_executions)
    if not pending:
        return
    done, pending = await asyncio.wait(pending, timeout=max(0.0, timeout_seconds))
    for task in done:
        _retained_executions.discard(task)
    if pending:
        logger.warning("Cancelling %s retained agent executions after shutdown grace", len(pending))
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
