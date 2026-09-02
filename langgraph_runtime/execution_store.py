"""Durable runtime execution and event storage.

The runtime keeps execution state here rather than in an SSE generator.  A
small in-memory implementation is retained for unit tests; production uses
PostgreSQL when AGENT_RUNTIME_EXECUTION_DATABASE_URL is configured.
"""

from __future__ import annotations

import asyncio
import json
import hashlib
import os
import uuid
from dataclasses import dataclass, field
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from langgraph_runtime.limits import required_positive_int


TERMINAL_STATUSES = frozenset({"completed", "clarification_required", "failed", "cancelled", "no_continuation"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_object(value: Any) -> dict[str, Any] | None:
    """Normalize asyncpg JSON values across codec/configuration variants."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str):
        decoded = json.loads(value)
        return dict(decoded) if isinstance(decoded, Mapping) else None
    return None


def _json_safe(value: Any) -> Any:
    """Convert runtime results to values accepted by PostgreSQL JSONB."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _event_row_to_dict(row: Any) -> dict[str, Any]:
    """Convert a PostgreSQL event row back to the complete wire envelope."""
    item = dict(row)
    item["payload"] = _json_object(item.get("payload")) or {}
    item["continuation"] = _json_object(item.get("continuation"))
    item["result"] = _json_object(item.get("result"))
    return item


def _event_id(run_id: str, attempt: int, event_id: str) -> str:
    prefix = f"{run_id}:attempt:{attempt}:"
    return event_id if event_id.startswith(prefix) else f"{prefix}{event_id}"


def request_fingerprint(operation: str, request: Mapping[str, Any]) -> str:
    """Fingerprint only execution semantics, excluding transport metadata."""
    value = {
        "operation": operation,
        "run_id": request.get("run_id"),
        "definition_id": request.get("definition_id"),
        "framework": request.get("framework"),
        "builder_id": request.get("builder_id"),
        "input": request.get("input") or {},
        "options": request.get("options") or {},
        "interrupt": request.get("interrupt") or {},
        "continuation": request.get("continuation"),
    }
    encoded = json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class ExecutionRecord:
    run_id: str
    operation: str
    request: dict[str, Any]
    payload: dict[str, Any]
    status: str = "queued"
    cancel_requested: bool = False
    next_sequence: int = 1
    attempt: int = 1
    continuation: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    owner_id: str | None = None
    lease_expires_at: str | None = None
    heartbeat_at: str | None = None
    fencing_token: int = 0
    request_fingerprint: str | None = None
    last_operation_id: str | None = None
    retry_source_attempt: int | None = None
    replay_only: bool = False


class LeaseLostError(RuntimeError):
    """Raised when a worker attempts to mutate a run it no longer owns."""


class ExecutionConflictError(RuntimeError):
    """Raised when an operation conflicts with an immutable terminal record."""


@dataclass(frozen=True)
class CancellationOutcome:
    """Atomic result of requesting cancellation for a durable execution."""

    outcome: str
    run_status: str | None = None

    @property
    def is_unknown(self) -> bool:
        return self.outcome == "unknown"

    @property
    def is_terminal(self) -> bool:
        return self.outcome == "terminal"

    @property
    def is_requested(self) -> bool:
        return self.outcome == "requested"


class ExecutionStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = os.getenv("AGENT_RUNTIME_EXECUTION_DATABASE_URL", "") if database_url is None else database_url
        self._records: dict[str, ExecutionRecord] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._operations: dict[tuple[str, str], dict[str, Any]] = {}
        self._condition = None
        self._lock = asyncio.Lock()
        self._pool = None
        self.owner_id = os.getenv("AGENT_RUNTIME_WORKER_ID") or f"runtime-{uuid.uuid4().hex}"
        self.lease_seconds = required_positive_int("AGENT_RUNTIME_LEASE_SECONDS")

    @property
    def durable(self) -> bool:
        return bool(self.database_url)

    async def initialize(self) -> None:
        if not self.database_url:
            return
        import asyncpg

        self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        async with self._pool.acquire() as connection:
            try:
                await connection.execute("select 1 from runtime_executions limit 0")
                await connection.execute("select 1 from runtime_operations limit 0")
                await connection.execute("select 1 from runtime_events limit 0")
            except asyncpg.UndefinedTableError as exc:
                raise RuntimeError(
                    "Runtime schema is missing; run app.db.migrate_runtime before starting the runtime"
                ) from exc

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def health(self) -> bool:
        if self._pool is None:
            return not self.database_url
        return bool(await self._pool.fetchval("select true"))

    async def create(
        self,
        run_id: str,
        operation: str,
        request: Mapping[str, Any],
        payload: Mapping[str, Any],
        *,
        operation_id: str | None = None,
        source_attempt: int | None = None,
    ) -> ExecutionRecord:
        fingerprint = request_fingerprint(operation, request)
        record = ExecutionRecord(run_id, operation, dict(request), dict(payload), request_fingerprint=fingerprint, last_operation_id=operation_id, retry_source_attempt=source_attempt)
        if self._pool is None:
            if operation_id:
                prior = self._operations.get((run_id, operation_id))
                if prior is not None:
                    if prior.get("fingerprint") != fingerprint:
                        raise ExecutionConflictError("operation_id was reused with different input")
                    existing = self._records.get(run_id)
                    if existing is None:
                        raise ExecutionConflictError("idempotent operation record has no execution")
                    return replace(existing, attempt=int(prior["attempt"]), replay_only=True)
            if run_id in self._records:
                existing = self._records[run_id]
                if operation == "retry":
                    if not operation_id:
                        raise ExecutionConflictError("retry requires operation_id")
                    prior = self._operations.get((run_id, operation_id))
                    if prior is not None:
                        return replace(existing, attempt=int(prior["attempt"]), replay_only=True)
                    if existing.status not in TERMINAL_STATUSES:
                        raise ExecutionConflictError("only terminal executions can be retried")
                    if source_attempt is None or source_attempt != existing.attempt:
                        raise ExecutionConflictError("retry source_attempt does not match the current terminal attempt")
                    existing.operation = str(request.get("retry_operation") or "start")
                    existing.request = dict(request.get("retry_request") or request)
                    existing.payload = dict(payload)
                    existing.status = "queued"
                    existing.cancel_requested = False
                    existing.attempt += 1
                    existing.continuation = None
                    existing.result = None
                    existing.error = None
                    existing.owner_id = None
                    existing.lease_expires_at = None
                    existing.heartbeat_at = None
                    existing.request_fingerprint = fingerprint
                    existing.last_operation_id = operation_id
                    existing.retry_source_attempt = source_attempt
                    self._operations[(run_id, operation_id)] = {"attempt": existing.attempt, "fingerprint": fingerprint}
                    return existing
                if existing.status in TERMINAL_STATUSES:
                    if existing.request_fingerprint in {None, fingerprint}:
                        return existing
                    raise ExecutionConflictError("terminal execution is immutable; use retry")
                if any(item.get("terminal") for item in self._events.get(run_id, [])):
                    return replace(existing, replay_only=True)
                return existing
            self._records[run_id] = record
            self._events.setdefault(run_id, [])
            if operation_id:
                self._operations[(run_id, operation_id)] = {
                    "attempt": record.attempt,
                    "fingerprint": fingerprint,
                    "status": record.status,
                    "result": None,
                }
            return record
        replay_attempt: int | None = None
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                if operation_id:
                    prior = await connection.fetchrow(
                        "select attempt, request_fingerprint from runtime_operations where run_id=$1 and operation_id=$2 for update",
                        run_id,
                        operation_id,
                    )
                    if prior is not None:
                        if str(prior["request_fingerprint"]) != fingerprint:
                            raise ExecutionConflictError("operation_id was reused with different input")
                        replay_attempt = int(prior["attempt"])
                existing = await connection.fetchrow(
                    "select operation, request, status, attempt, continuation, request_fingerprint from runtime_executions where run_id=$1 for update",
                    run_id,
                )
                if existing is not None and operation == "retry":
                    if not operation_id:
                        raise ExecutionConflictError("retry requires operation_id")
                    prior = await connection.fetchrow("select attempt from runtime_operations where run_id=$1 and operation_id=$2", run_id, operation_id)
                    if prior is not None:
                        replay_attempt = int(prior["attempt"])
                    elif existing["status"] not in TERMINAL_STATUSES:
                        raise ExecutionConflictError("only terminal executions can be retried")
                    elif source_attempt is None or source_attempt != existing["attempt"]:
                        raise ExecutionConflictError("retry source_attempt does not match the current terminal attempt")
                    else:
                        await connection.execute(
                            """update runtime_executions
                           set operation=$2, request=$3::jsonb, payload=$4::jsonb,
                               status='queued', cancel_requested=false, attempt=attempt+1,
                               continuation=null, result=null, error=null, owner_id=null,
                               lease_expires_at=null, heartbeat_at=null, updated_at=now(),
                               request_fingerprint=$5, last_operation_id=$6, retry_source_attempt=$7
                           where run_id=$1""",
                            run_id, str(request.get("retry_operation") or "start"), json.dumps(_json_safe(dict(request.get("retry_request") or request))), json.dumps(_json_safe(dict(payload))), fingerprint, operation_id, source_attempt,
                        )
                        await connection.execute("insert into runtime_operations(run_id, operation_id, operation, request_fingerprint, attempt) values($1,$2,$3,$4,(select attempt from runtime_executions where run_id=$1))", run_id, operation_id, operation, fingerprint)
                elif existing is not None and existing["status"] in TERMINAL_STATUSES:
                    existing_fingerprint = existing["request_fingerprint"] or request_fingerprint(existing["operation"], _json_object(existing["request"]) or {})
                    if existing_fingerprint != fingerprint:
                        raise ExecutionConflictError("terminal execution is immutable; use retry")
                elif existing is None:
                    await connection.execute(
                        """insert into runtime_executions(run_id, operation, request, payload, status, request_fingerprint, last_operation_id, retry_source_attempt)
                           values($1,$2,$3::jsonb,$4::jsonb,'queued',$5,$6,$7)
                           on conflict (run_id) do nothing""",
                        run_id, operation, json.dumps(_json_safe(dict(request))), json.dumps(_json_safe(dict(payload))), fingerprint, operation_id, source_attempt,
                    )
                if operation_id and replay_attempt is None:
                    await connection.execute(
                        """insert into runtime_operations(run_id, operation_id, operation, request_fingerprint, attempt, status)
                           values($1,$2,$3,$4,(select attempt from runtime_executions where run_id=$1),'queued')
                           on conflict (run_id, operation_id) do nothing""",
                        run_id,
                        operation_id,
                        operation,
                        fingerprint,
                    )
        current = await self.get(run_id)
        if current is None:
            return current  # type: ignore[return-value]
        if replay_attempt is not None:
            return replace(current, attempt=replay_attempt, replay_only=True)
        if await self._pool.fetchval(
            "select exists(select 1 from runtime_events where run_id=$1 and terminal=true)",
            run_id,
        ):
            return replace(current, replay_only=True)
        return current

    async def get(self, run_id: str) -> ExecutionRecord | None:
        if self._pool is None:
            return self._records.get(run_id)
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow("select * from runtime_executions where run_id=$1", run_id)
        if row is None:
            return None
        return ExecutionRecord(
            run_id=row["run_id"], operation=row["operation"],
            request=_json_object(row["request"]) or {}, payload=_json_object(row["payload"]) or {},
            status=row["status"], cancel_requested=row["cancel_requested"], next_sequence=row["next_sequence"], attempt=row["attempt"],
            continuation=_json_object(row["continuation"]),
            result=_json_object(row["result"]), error=_json_object(row["error"]),
            created_at=row["created_at"].isoformat(), updated_at=row["updated_at"].isoformat(),
            owner_id=row.get("owner_id") if hasattr(row, "get") else row["owner_id"],
            lease_expires_at=row["lease_expires_at"].isoformat() if row.get("lease_expires_at") else None,
            heartbeat_at=row["heartbeat_at"].isoformat() if row.get("heartbeat_at") else None,
            fencing_token=int(row.get("fencing_token") or 0),
            request_fingerprint=row.get("request_fingerprint") if hasattr(row, "get") else None,
            last_operation_id=row.get("last_operation_id") if hasattr(row, "get") else None,
            retry_source_attempt=row.get("retry_source_attempt") if hasattr(row, "get") else None,
        )

    async def claim(self, run_id: str, *, owner_id: str | None = None, lease_seconds: int | None = None) -> int | None:
        owner_id = owner_id or self.owner_id
        lease_seconds = lease_seconds or self.lease_seconds
        if self._pool is None:
            record = self._records.get(run_id)
            if record is None or record.status in TERMINAL_STATUSES:
                return None
            now = datetime.now(timezone.utc)
            if record.lease_expires_at:
                expires = datetime.fromisoformat(record.lease_expires_at)
                if expires > now and record.owner_id != owner_id:
                    return None
            record.owner_id = owner_id
            record.lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
            record.heartbeat_at = now.isoformat()
            record.fencing_token += 1
            return record.fencing_token
        async with self._pool.acquire() as connection:
            row = await connection.fetchrow(
                """update runtime_executions
                   set owner_id=$2, lease_expires_at=now() + ($3 * interval '1 second'),
                       heartbeat_at=now(), fencing_token=fencing_token+1, updated_at=now()
                   where run_id=$1
                     and status not in ('completed','clarification_required','failed','cancelled','no_continuation')
                     and (lease_expires_at is null or lease_expires_at < now() or owner_id=$2)
                   returning fencing_token""",
                run_id, owner_id, lease_seconds,
            )
            return int(row["fencing_token"]) if row else None

    async def heartbeat(self, run_id: str, *, owner_id: str, fencing_token: int, lease_seconds: int | None = None) -> bool:
        lease_seconds = lease_seconds or self.lease_seconds
        if self._pool is None:
            record = self._records.get(run_id)
            if not record or record.owner_id != owner_id or record.fencing_token != fencing_token:
                return False
            now = datetime.now(timezone.utc)
            record.heartbeat_at = now.isoformat()
            record.lease_expires_at = (now + timedelta(seconds=lease_seconds)).isoformat()
            return True
        result = await self._pool.execute(
            """update runtime_executions set heartbeat_at=now(), lease_expires_at=now() + ($4 * interval '1 second'), updated_at=now()
               where run_id=$1 and owner_id=$2 and fencing_token=$3 and status not in ('completed','clarification_required','failed','cancelled','no_continuation')""",
            run_id, owner_id, fencing_token, lease_seconds,
        )
        return result.endswith("1")

    async def set_status(self, run_id: str, status: str, *, result: Mapping[str, Any] | None = None, error: Mapping[str, Any] | None = None, owner_id: str | None = None, fencing_token: int | None = None) -> None:
        if self._pool is None:
            record = self._records[run_id]
            if owner_id is not None and (record.owner_id != owner_id or record.fencing_token != fencing_token):
                raise LeaseLostError(f"lost runtime lease for {run_id}")
            record.status, record.result, record.error, record.updated_at = status, dict(result) if result else None, dict(error) if error else None, _now()
            if record.last_operation_id:
                operation_record = self._operations.get((run_id, record.last_operation_id))
                if operation_record is not None:
                    operation_record["status"] = status
                    operation_record["result"] = dict(result) if result else None
            if status in TERMINAL_STATUSES:
                record.cancel_requested = False
            return
        if owner_id is None or fencing_token is None:
            raise LeaseLostError(f"runtime status mutation requires a lease for {run_id}")
        result_status = await self._pool.execute(
            """update runtime_executions set status=$2, cancel_requested=case when $2 in ('completed','clarification_required','failed','cancelled','no_continuation') then false else cancel_requested end,
                                                   result=coalesce($3::jsonb,result), error=coalesce($4::jsonb,error), updated_at=now()
               where run_id=$1 and owner_id=$5 and fencing_token=$6 and (lease_expires_at is null or lease_expires_at > now())""",
            run_id, status, json.dumps(_json_safe(dict(result))) if result else None, json.dumps(_json_safe(dict(error))) if error else None, owner_id, fencing_token,
        )
        if not result_status.endswith("1"):
            raise LeaseLostError(f"lost runtime lease for {run_id}")
        await self._pool.execute(
            """update runtime_operations set status=$2, result=coalesce($3::jsonb, result)
               where run_id=$1 and operation_id=(select last_operation_id from runtime_executions where run_id=$1)""",
            run_id,
            status,
            json.dumps(_json_safe(dict(result))) if result else None,
        )

    async def request_cancel(self, run_id: str) -> CancellationOutcome:
        if self._pool is None:
            record = self._records.get(run_id)
            if record is None:
                return CancellationOutcome("unknown")
            if record.status in TERMINAL_STATUSES:
                return CancellationOutcome("terminal", record.status)
            if record.status in {"awaiting_human", "paused"}:
                record.status = "cancelled"
                record.cancel_requested = False
                record.updated_at = _now()
                return CancellationOutcome("terminal", record.status)
            record.cancel_requested = True
            return CancellationOutcome("requested", record.status)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                record = await connection.fetchrow(
                    "select status, cancel_requested from runtime_executions where run_id=$1 for update",
                    run_id,
                )
                if record is None:
                    return CancellationOutcome("unknown")
                status = str(record["status"])
                if status in TERMINAL_STATUSES:
                    return CancellationOutcome("terminal", status)
                if status in {"awaiting_human", "paused"}:
                    await connection.execute(
                        "update runtime_executions set status='cancelled', cancel_requested=false, updated_at=now() where run_id=$1",
                        run_id,
                    )
                    return CancellationOutcome("terminal", "cancelled")
                await connection.execute(
                    "update runtime_executions set cancel_requested=true, updated_at=now() where run_id=$1 and status not in ('completed','clarification_required','failed','cancelled','no_continuation')",
                    run_id,
                )
                return CancellationOutcome("requested", status)

    async def is_cancel_requested(self, run_id: str) -> bool:
        record = await self.get(run_id)
        return bool(record and record.cancel_requested)

    async def request_pause(self, run_id: str) -> dict[str, Any]:
        """Persist a cooperative pause request without changing execution ownership."""
        if self._pool is None:
            record = self._records.get(run_id)
            if record is None:
                return {"status": "unknown", "run_id": run_id}
            if record.status in TERMINAL_STATUSES:
                return {"status": "terminal", "run_id": run_id, "run_status": record.status}
            if record.status in {"awaiting_human", "paused"}:
                return {"status": "already_paused", "run_id": run_id, "run_status": record.status}
            record.payload["pause_requested"] = True
            record.updated_at = _now()
            return {"status": "pause_requested", "run_id": run_id, "run_status": record.status}
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    "select status, payload from runtime_executions where run_id=$1 for update",
                    run_id,
                )
                if row is None:
                    return {"status": "unknown", "run_id": run_id}
                status = str(row["status"])
                if status in TERMINAL_STATUSES:
                    return {"status": "terminal", "run_id": run_id, "run_status": status}
                if status in {"awaiting_human", "paused"}:
                    return {"status": "already_paused", "run_id": run_id, "run_status": status}
                await connection.execute(
                    "update runtime_executions set payload=payload || '{\"pause_requested\": true}'::jsonb, updated_at=now() where run_id=$1",
                    run_id,
                )
                return {"status": "pause_requested", "run_id": run_id, "run_status": status}

    async def is_pause_requested(self, run_id: str) -> bool:
        record = await self.get(run_id)
        return bool(record and record.payload.get("pause_requested") is True)

    async def clear_pause_request(self, run_id: str) -> None:
        if self._pool is None:
            record = self._records.get(run_id)
            if record is not None:
                record.payload.pop("pause_requested", None)
                record.updated_at = _now()
            return
        await self._pool.execute(
            "update runtime_executions set payload=payload - 'pause_requested', updated_at=now() where run_id=$1",
            run_id,
        )

    @staticmethod
    def _correction_fingerprint(correction: Mapping[str, Any]) -> str:
        encoded = json.dumps(_json_safe(dict(correction)), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    async def request_course_correction(
        self,
        run_id: str,
        correction: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Atomically queue one correction and its replayable acceptance event."""

        value = dict(correction)
        operation_id = str(value["operation_id"])
        correction_id = str(value["correction_id"])
        fingerprint = self._correction_fingerprint(value)

        if self._pool is None:
            async with self._lock:
                record = self._records.get(run_id)
                if record is None:
                    return {"status": "unknown", "run_id": run_id}
                if record.status in TERMINAL_STATUSES:
                    return {
                        "status": "terminal", "run_id": run_id,
                        "run_status": record.status, "correction_id": correction_id,
                        "operation_id": operation_id,
                    }
                prior = self._operations.get((run_id, operation_id))
                if prior is not None:
                    if prior.get("fingerprint") != fingerprint:
                        raise ExecutionConflictError("operation_id was reused with different correction input")
                    receipt = dict(prior["result"])
                    if receipt.get("status") == "accepted":
                        receipt["status"] = "already_accepted"
                    return receipt
                queued = {**value, "status": "accepted"}
                corrections = list(record.payload.get("course_corrections") or [])
                corrections.append(queued)
                record.payload["course_corrections"] = corrections
                receipt = {
                    "status": "accepted", "run_id": run_id,
                    "run_status": record.status, "correction_id": correction_id,
                    "operation_id": operation_id,
                }
                self._operations[(run_id, operation_id)] = {
                    "attempt": record.attempt, "fingerprint": fingerprint,
                    "status": "accepted", "result": receipt,
                }
                event = {
                    "run_id": run_id, "sequence": record.next_sequence,
                    "attempt": record.attempt,
                    "event_id": _event_id(run_id, record.attempt, f"course-correction:{correction_id}:accepted"),
                    "kind": "course_correction.accepted", "occurred_at": _now(),
                    "payload": {"correction_id": correction_id, "operation_id": operation_id},
                    "terminal": False, "result": None,
                }
                record.next_sequence += 1
                record.updated_at = _now()
                self._events.setdefault(run_id, []).append(event)
                return receipt

        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    "select status, payload, next_sequence, attempt from runtime_executions where run_id=$1 for update",
                    run_id,
                )
                if row is None:
                    return {"status": "unknown", "run_id": run_id}
                status = str(row["status"])
                if status in TERMINAL_STATUSES:
                    return {
                        "status": "terminal", "run_id": run_id,
                        "run_status": status, "correction_id": correction_id,
                        "operation_id": operation_id,
                    }
                prior = await connection.fetchrow(
                    "select request_fingerprint, result from runtime_operations where run_id=$1 and operation_id=$2",
                    run_id, operation_id,
                )
                if prior is not None:
                    if str(prior["request_fingerprint"]) != fingerprint:
                        raise ExecutionConflictError("operation_id was reused with different correction input")
                    receipt = _json_object(prior["result"]) or {}
                    receipt["status"] = "already_accepted" if receipt.get("status") == "accepted" else receipt.get("status")
                    return receipt
                payload = _json_object(row["payload"]) or {}
                corrections = list(payload.get("course_corrections") or [])
                corrections.append({**value, "status": "accepted"})
                payload["course_corrections"] = corrections
                receipt = {
                    "status": "accepted", "run_id": run_id, "run_status": status,
                    "correction_id": correction_id, "operation_id": operation_id,
                }
                sequence = int(row["next_sequence"])
                attempt = int(row["attempt"])
                await connection.execute(
                    "update runtime_executions set payload=$2::jsonb, next_sequence=$3, updated_at=now() where run_id=$1",
                    run_id, json.dumps(payload), sequence + 1,
                )
                await connection.execute(
                    "insert into runtime_operations(run_id, operation_id, operation, request_fingerprint, attempt, status, result) values($1,$2,'course_correction',$3,$4,'accepted',$5::jsonb)",
                    run_id, operation_id, fingerprint, attempt, json.dumps(receipt),
                )
                await connection.execute(
                    """insert into runtime_events(run_id, sequence, attempt, event_id, kind, payload, occurred_at, terminal)
                       values($1,$2,$3,$4,'course_correction.accepted',$5::jsonb,$6,false)""",
                    run_id, sequence, attempt,
                    _event_id(run_id, attempt, f"course-correction:{correction_id}:accepted"),
                    json.dumps({"correction_id": correction_id, "operation_id": operation_id}), _now(),
                )
                return receipt

    async def pending_course_corrections(self, run_id: str) -> list[dict[str, Any]]:
        record = await self.get(run_id)
        if record is None:
            return []
        return [
            dict(value) for value in record.payload.get("course_corrections") or []
            if isinstance(value, Mapping) and value.get("status") == "accepted"
        ]

    async def mark_course_corrections_applied(
        self,
        run_id: str,
        correction_ids: list[str],
        *,
        plan_revision: int,
    ) -> list[str]:
        """Atomically acknowledge applied corrections and append one replay event."""

        selected = {str(value) for value in correction_ids if value}
        if not selected:
            return []
        if self._pool is None:
            async with self._lock:
                record = self._records.get(run_id)
                if record is None:
                    return []
                applied: list[str] = []
                corrections = list(record.payload.get("course_corrections") or [])
                for value in corrections:
                    if isinstance(value, dict) and value.get("status") == "accepted" and str(value.get("correction_id")) in selected:
                        value.update({"status": "applied", "plan_revision": int(plan_revision)})
                        applied.append(str(value["correction_id"]))
                        operation = self._operations.get((run_id, str(value.get("operation_id"))))
                        if operation is not None:
                            operation.update({"status": "completed", "result": {
                                **dict(operation.get("result") or {}), "status": "applied",
                                "plan_revision": int(plan_revision),
                            }})
                if not applied:
                    return []
                record.payload["course_corrections"] = corrections
                self._events.setdefault(run_id, []).append({
                    "run_id": run_id, "sequence": record.next_sequence, "attempt": record.attempt,
                    "event_id": _event_id(run_id, record.attempt, f"course-correction:{'-'.join(sorted(applied))}:applied"),
                    "kind": "course_correction.applied", "occurred_at": _now(),
                    "payload": {"correction_ids": sorted(applied), "plan_revision": int(plan_revision)},
                    "terminal": False, "result": None,
                })
                record.next_sequence += 1
                record.updated_at = _now()
                return sorted(applied)

        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    "select payload, next_sequence, attempt from runtime_executions where run_id=$1 for update",
                    run_id,
                )
                if row is None:
                    return []
                payload = _json_object(row["payload"]) or {}
                corrections = list(payload.get("course_corrections") or [])
                applied: list[str] = []
                operation_ids: list[str] = []
                for value in corrections:
                    if isinstance(value, dict) and value.get("status") == "accepted" and str(value.get("correction_id")) in selected:
                        value.update({"status": "applied", "plan_revision": int(plan_revision)})
                        applied.append(str(value["correction_id"]))
                        operation_ids.append(str(value.get("operation_id") or ""))
                if not applied:
                    return []
                payload["course_corrections"] = corrections
                sequence, attempt = int(row["next_sequence"]), int(row["attempt"])
                await connection.execute(
                    "update runtime_executions set payload=$2::jsonb, next_sequence=$3, updated_at=now() where run_id=$1",
                    run_id, json.dumps(payload), sequence + 1,
                )
                for operation_id in operation_ids:
                    if operation_id:
                        await connection.execute(
                            "update runtime_operations set status='completed', result=coalesce(result,'{}'::jsonb) || $3::jsonb where run_id=$1 and operation_id=$2",
                            run_id, operation_id,
                            json.dumps({"status": "applied", "plan_revision": int(plan_revision)}),
                        )
                await connection.execute(
                    """insert into runtime_events(run_id, sequence, attempt, event_id, kind, payload, occurred_at, terminal)
                       values($1,$2,$3,$4,'course_correction.applied',$5::jsonb,$6,false)""",
                    run_id, sequence, attempt,
                    _event_id(run_id, attempt, f"course-correction:{'-'.join(sorted(applied))}:applied"),
                    json.dumps({"correction_ids": sorted(applied), "plan_revision": int(plan_revision)}), _now(),
                )
                return sorted(applied)

    async def append(self, run_id: str, event: Mapping[str, Any], result: Mapping[str, Any] | None = None, *, attempt: int | None = None, owner_id: str | None = None, fencing_token: int | None = None) -> dict[str, Any]:
        item = dict(event)
        if self._pool is None:
            record = self._records[run_id]
            if owner_id is not None and (record.owner_id != owner_id or record.fencing_token != fencing_token):
                raise LeaseLostError(f"lost runtime lease for {run_id}")
            attempt = attempt or record.attempt
            item["attempt"] = attempt
            item["event_id"] = _event_id(run_id, attempt, item["event_id"])
            existing = next((value for value in self._events.setdefault(run_id, []) if value["event_id"] == item["event_id"]), None)
            if existing is not None:
                return existing
            item["sequence"] = record.next_sequence
            record.next_sequence += 1
            item["result"] = dict(result) if result else None
            self._events[run_id].append(item)
            if item.get("continuation") is not None:
                record.continuation = dict(item["continuation"])
            return item
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """select next_sequence, attempt from runtime_executions
                       where run_id=$1 and owner_id=$2 and fencing_token=$3
                         and (lease_expires_at is null or lease_expires_at > now()) for update""",
                    run_id, owner_id, fencing_token,
                )
                if row is None:
                    raise LeaseLostError(f"lost runtime lease for {run_id}")
                attempt = attempt or int(row["attempt"])
                sequence = int(row["next_sequence"])
                item["sequence"] = sequence
                item["attempt"] = attempt
                item["event_id"] = _event_id(run_id, attempt, item["event_id"])
                continuation = item.get("continuation")
                inserted = await connection.fetchrow(
                    """insert into runtime_events(
                           run_id, sequence, attempt, event_id, kind, payload, occurred_at,
                           trace_id,
                           continuation, terminal, result
                       ) values($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9::jsonb,$10,$11::jsonb)
                       on conflict (run_id, event_id) do nothing
                       returning *""",
                    run_id,
                    sequence,
                    item["attempt"],
                    item["event_id"],
                    item.get("kind", "runtime.event"),
                    json.dumps(_json_safe(item.get("payload") or {})),
                    item.get("occurred_at"),
                    item.get("trace_id"),
                    json.dumps(_json_safe(continuation)) if continuation is not None else None,
                    bool(item.get("terminal")),
                    json.dumps(_json_safe(dict(result))) if result else None,
                )
                if inserted is None:
                    existing = await connection.fetchrow(
                        "select * from runtime_events where run_id=$1 and event_id=$2",
                        run_id,
                        item["event_id"],
                    )
                    return _event_row_to_dict(existing)

                if continuation is not None:
                    await connection.execute(
                        "update runtime_executions set continuation=$2::jsonb, next_sequence=next_sequence+1, updated_at=now() where run_id=$1",
                        run_id,
                        json.dumps(_json_safe(continuation)),
                    )
                else:
                    await connection.execute(
                        "update runtime_executions set next_sequence=next_sequence+1, updated_at=now() where run_id=$1",
                        run_id,
                    )
        return item

    async def set_event_result(
        self,
        run_id: str,
        event_id: str,
        result: Mapping[str, Any],
        *,
        owner_id: str | None = None,
        fencing_token: int | None = None,
    ) -> None:
        """Attach the sole terminal result to an already-journaled terminal event."""
        if self._pool is None:
            record = self._records[run_id]
            if owner_id is not None and (record.owner_id != owner_id or record.fencing_token != fencing_token):
                raise LeaseLostError(f"lost runtime lease for {run_id}")
            for item in self._events.get(run_id, []):
                if item.get("event_id") == event_id:
                    if item.get("result") not in (None, dict(result)):
                        raise ExecutionConflictError("terminal event already has a conflicting result")
                    item["result"] = dict(result)
                    return
            raise ExecutionConflictError("terminal event was not found")
        async with self._pool.acquire() as connection:
            updated = await connection.execute(
                """update runtime_events as events set result=$3::jsonb
                   where events.run_id=$1 and events.event_id=$2 and events.terminal=true
                     and exists (
                       select 1 from runtime_executions executions
                       where executions.run_id=events.run_id
                         and executions.owner_id=$4 and executions.fencing_token=$5
                         and (executions.lease_expires_at is null or executions.lease_expires_at > now())
                     )""",
                run_id,
                event_id,
                json.dumps(_json_safe(dict(result))),
                owner_id,
                fencing_token,
            )
            if not updated.endswith("1"):
                raise ExecutionConflictError("terminal event was not found")

    async def finalize_execution(
        self,
        run_id: str,
        terminal_event: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        status: str,
        error: Mapping[str, Any] | None = None,
        attempt: int | None = None,
        owner_id: str,
        fencing_token: int,
    ) -> dict[str, Any]:
        """Atomically commit the terminal journal entry and execution state."""
        if status not in TERMINAL_STATUSES:
            raise ValueError("runtime finalization requires a terminal status")
        item = dict(terminal_event)
        if not bool(item.get("terminal")):
            raise ValueError("runtime finalization requires a terminal event")
        safe_result = _json_safe(dict(result))
        safe_error = _json_safe(dict(error)) if error else None
        if self._pool is None:
            async with self._lock:
                record = self._records[run_id]
                if record.owner_id != owner_id or record.fencing_token != fencing_token:
                    raise LeaseLostError(f"lost runtime lease for {run_id}")
                effective_attempt = attempt or record.attempt
                terminals = [value for value in self._events.get(run_id, []) if value.get("terminal") and int(value.get("attempt", 1)) == effective_attempt]
                if len(terminals) > 1:
                    raise ExecutionConflictError("multiple terminal events were persisted")
                if terminals:
                    stored = terminals[0]
                    if stored.get("result") not in (None, safe_result):
                        raise ExecutionConflictError("terminal event already has a conflicting result")
                    stored["result"] = safe_result
                else:
                    item["attempt"] = effective_attempt
                    item["event_id"] = _event_id(run_id, effective_attempt, str(item["event_id"]))
                    occupied = {
                        int(value.get("sequence", 0))
                        for value in self._events.get(run_id, [])
                    }
                    sequence = record.next_sequence
                    while sequence in occupied:
                        sequence += 1
                    item["sequence"] = sequence
                    item["result"] = safe_result
                    self._events.setdefault(run_id, []).append(item)
                    record.next_sequence = max(record.next_sequence, sequence + 1)
                    stored = item
                record.status = status
                record.result = dict(safe_result)
                record.error = dict(safe_error) if safe_error else None
                if item.get("continuation") is not None:
                    record.continuation = dict(item["continuation"])
                record.cancel_requested = False
                record.owner_id = None
                record.lease_expires_at = None
                record.heartbeat_at = None
                record.updated_at = _now()
                if record.last_operation_id:
                    operation = self._operations.get((run_id, record.last_operation_id))
                    if operation is not None:
                        operation["status"] = status
                        operation["result"] = dict(safe_result)
                return dict(stored)

        async with self._pool.acquire() as connection:
            async with connection.transaction():
                record = await connection.fetchrow(
                    """select attempt, next_sequence, last_operation_id from runtime_executions
                       where run_id=$1 and owner_id=$2 and fencing_token=$3
                         and (lease_expires_at is null or lease_expires_at > now()) for update""",
                    run_id, owner_id, fencing_token,
                )
                if record is None:
                    raise LeaseLostError(f"lost runtime lease for {run_id}")
                effective_attempt = attempt or int(record["attempt"])
                terminals = await connection.fetch(
                    "select * from runtime_events where run_id=$1 and attempt=$2 and terminal=true for update",
                    run_id, effective_attempt,
                )
                if len(terminals) > 1:
                    raise ExecutionConflictError("multiple terminal events were persisted")
                if terminals:
                    existing_result = _json_object(terminals[0]["result"])
                    if existing_result not in (None, safe_result):
                        raise ExecutionConflictError("terminal event already has a conflicting result")
                    stored = await connection.fetchrow(
                        "update runtime_events set result=$3::jsonb where run_id=$1 and event_id=$2 returning *",
                        run_id, terminals[0]["event_id"], json.dumps(safe_result),
                    )
                else:
                    sequence = int(record["next_sequence"])
                    existing_sequence = await connection.fetchrow(
                        "select sequence from runtime_events where run_id=$1 and sequence=$2 for update",
                        run_id, sequence,
                    )
                    if existing_sequence is not None:
                        # Older runtime versions could commit a resumable
                        # event without advancing next_sequence. Recovery may
                        # then be finalizing a cancellation or failure while
                        # that event is already occupying the next slot.
                        sequence = int(existing_sequence["sequence"]) + 1
                    event_id = _event_id(run_id, effective_attempt, str(item["event_id"]))
                    stored = await connection.fetchrow(
                        """insert into runtime_events(
                               run_id, sequence, attempt, event_id, kind, payload, occurred_at,
                               trace_id, continuation, terminal, result
                           ) values($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9::jsonb,true,$10::jsonb)
                           returning *""",
                        run_id, sequence, effective_attempt, event_id,
                        item.get("kind", "run.completed"), json.dumps(_json_safe(item.get("payload") or {})),
                        item.get("occurred_at"), item.get("trace_id"),
                        json.dumps(_json_safe(item.get("continuation"))) if item.get("continuation") is not None else None,
                        json.dumps(safe_result),
                    )
                    await connection.execute(
                        "update runtime_executions set next_sequence=greatest(next_sequence, $2) where run_id=$1",
                        run_id, sequence + 1,
                    )
                await connection.execute(
                    """update runtime_executions
                       set status=$2, result=$3::jsonb, error=$4::jsonb,
                           continuation=coalesce($5::jsonb, continuation), cancel_requested=false,
                           owner_id=null, lease_expires_at=null, heartbeat_at=null, updated_at=now()
                       where run_id=$1""",
                    run_id, status, json.dumps(safe_result), json.dumps(safe_error) if safe_error else None,
                    json.dumps(_json_safe(item.get("continuation"))) if item.get("continuation") is not None else None,
                )
                if record["last_operation_id"]:
                    await connection.execute(
                        "update runtime_operations set status=$3, result=$4::jsonb where run_id=$1 and operation_id=$2",
                        run_id, record["last_operation_id"], status, json.dumps(safe_result),
                    )
                return _event_row_to_dict(stored)

    async def checkpoint_execution(
        self,
        run_id: str,
        checkpoint_event: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        status: str,
        continuation: Mapping[str, Any] | None,
        attempt: int | None = None,
        owner_id: str,
        fencing_token: int,
    ) -> dict[str, Any]:
        """Persist a resumable checkpoint and release the runtime lease."""
        if status not in {"awaiting_human", "paused"}:
            raise ValueError("runtime checkpoint requires a resumable status")
        item = dict(checkpoint_event)
        if bool(item.get("terminal")):
            raise ValueError("runtime checkpoint event must be nonterminal")
        safe_result = _json_safe(dict(result))
        safe_continuation = _json_safe(dict(continuation)) if continuation else None
        if self._pool is None:
            async with self._lock:
                record = self._records[run_id]
                if record.owner_id != owner_id or record.fencing_token != fencing_token:
                    raise LeaseLostError(f"lost runtime lease for {run_id}")
                effective_attempt = attempt or record.attempt
                item["attempt"] = effective_attempt
                event_id = _event_id(run_id, effective_attempt, str(item["event_id"]))
                existing = next(
                    (value for value in self._events.setdefault(run_id, []) if value.get("event_id") == event_id),
                    None,
                )
                if existing is not None:
                    record.status = status
                    record.result = dict(safe_result)
                    record.continuation = dict(safe_continuation) if safe_continuation else None
                    record.owner_id = record.lease_expires_at = record.heartbeat_at = None
                    record.updated_at = _now()
                    return dict(existing)
                item["event_id"] = event_id
                item["sequence"] = record.next_sequence
                item["result"] = safe_result
                item["continuation"] = safe_continuation
                self._events.setdefault(run_id, []).append(item)
                record.next_sequence += 1
                record.status = status
                record.result = dict(safe_result)
                record.continuation = dict(safe_continuation) if safe_continuation else None
                record.owner_id = record.lease_expires_at = record.heartbeat_at = None
                record.updated_at = _now()
                return dict(item)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                record = await connection.fetchrow(
                    """select attempt, next_sequence, last_operation_id from runtime_executions
                       where run_id=$1 and owner_id=$2 and fencing_token=$3
                         and (lease_expires_at is null or lease_expires_at > now()) for update""",
                    run_id, owner_id, fencing_token,
                )
                if record is None:
                    raise LeaseLostError(f"lost runtime lease for {run_id}")
                effective_attempt = attempt or int(record["attempt"])
                event_id = _event_id(run_id, effective_attempt, str(item["event_id"]))
                # A worker can lose its lease after inserting the checkpoint
                # event but before the caller observes completion. Recovery
                # must finish that checkpoint instead of inserting the same
                # event at the stale next_sequence again.
                existing = await connection.fetchrow(
                    "select * from runtime_events where run_id=$1 and event_id=$2 for update",
                    run_id, event_id,
                )
                if existing is not None:
                    await connection.execute(
                        """update runtime_executions
                           set status=$2, result=$3::jsonb, continuation=$4::jsonb,
                               next_sequence=greatest(next_sequence, $5),
                               owner_id=null, lease_expires_at=null, heartbeat_at=null, updated_at=now()
                           where run_id=$1""",
                        run_id, status, json.dumps(safe_result),
                        json.dumps(safe_continuation) if safe_continuation else None,
                        int(existing["sequence"]) + 1,
                    )
                    if record["last_operation_id"]:
                        await connection.execute(
                            "update runtime_operations set status=$3, result=$4::jsonb where run_id=$1 and operation_id=$2",
                            run_id, record["last_operation_id"], status, json.dumps(safe_result),
                        )
                    return _event_row_to_dict(existing)
                stored = await connection.fetchrow(
                    """insert into runtime_events(
                           run_id, sequence, attempt, event_id, kind, payload, occurred_at,
                           trace_id, continuation, terminal, result
                       ) values($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9::jsonb,false,$10::jsonb)
                       returning *""",
                    run_id, int(record["next_sequence"]), effective_attempt, event_id,
                    item.get("kind", "run.paused"), json.dumps(_json_safe(item.get("payload") or {})),
                    item.get("occurred_at"), item.get("trace_id"),
                    json.dumps(safe_continuation) if safe_continuation else None,
                    json.dumps(safe_result),
                )
                await connection.execute(
                    """update runtime_executions
                       set status=$2, result=$3::jsonb, continuation=$4::jsonb,
                           next_sequence=next_sequence+1,
                           owner_id=null, lease_expires_at=null, heartbeat_at=null, updated_at=now()
                       where run_id=$1""",
                    run_id, status, json.dumps(safe_result),
                    json.dumps(safe_continuation) if safe_continuation else None,
                )
                if record["last_operation_id"]:
                    await connection.execute(
                        "update runtime_operations set status=$3, result=$4::jsonb where run_id=$1 and operation_id=$2",
                        run_id, record["last_operation_id"], status, json.dumps(safe_result),
                    )
                return _event_row_to_dict(stored)

    async def events_after(self, run_id: str, sequence: int = 0, *, attempt: int | None = None) -> list[dict[str, Any]]:
        if attempt is None:
            record = await self.get(run_id)
            attempt = record.attempt if record else None
        if self._pool is None:
            return [dict(item) for item in self._events.get(run_id, []) if int(item.get("sequence", 0)) > sequence and (attempt is None or int(item.get("attempt", 1)) == attempt)]
        rows = await self._pool.fetch("select * from runtime_events where run_id=$1 and sequence>$2 and ($3::integer is null or attempt=$3) order by sequence", run_id, sequence, attempt)
        events = []
        for row in rows:
            events.append(_event_row_to_dict(row))
        return events

    async def nonterminal(self) -> list[ExecutionRecord]:
        return await self.list_recovery_candidates(limit=100)

    async def list_recovery_candidates(self, limit: int = 100) -> list[ExecutionRecord]:
        """Return bounded nonterminal records for repeated lease recovery."""
        limit = max(1, min(int(limit), 1000))
        if self._pool is None:
            terminal_run_ids = {run_id for run_id, events in self._events.items() if any(item.get("terminal") for item in events)}
            records = [record for record in self._records.values() if record.status not in TERMINAL_STATUSES and record.run_id not in terminal_run_ids]
            now = datetime.now(timezone.utc)
            def recovery_key(record: ExecutionRecord) -> tuple[int, str, str]:
                expired = 0
                if record.lease_expires_at:
                    expired = 0 if datetime.fromisoformat(record.lease_expires_at) <= now else 1
                return expired, record.updated_at, record.run_id
            return sorted(records, key=recovery_key)[:limit]
        rows = await self._pool.fetch(
            """select executions.run_id from runtime_executions executions
               where executions.status not in ('completed','clarification_required','failed','cancelled','no_continuation')
                 and not exists (select 1 from runtime_events events where events.run_id=executions.run_id and events.terminal=true)
               order by case when lease_expires_at is not null and lease_expires_at < now() then 0 else 1 end,
                        executions.updated_at, executions.run_id
               limit $1""",
            limit,
        )
        return [record for row in rows if (record := await self.get(row["run_id"])) is not None]

    async def list_terminal_reconciliation_candidates(self, limit: int = 100) -> list[ExecutionRecord]:
        limit = max(1, min(int(limit), 1000))
        if self._pool is None:
            run_ids = [run_id for run_id, events in self._events.items() if any(item.get("terminal") for item in events)]
            return [self._records[run_id] for run_id in run_ids if self._records[run_id].status not in TERMINAL_STATUSES][:limit]
        rows = await self._pool.fetch(
            """select distinct executions.run_id from runtime_executions executions
               join runtime_events events on events.run_id=executions.run_id and events.terminal=true
               where executions.status not in ('completed','clarification_required','failed','cancelled','no_continuation')
               order by executions.run_id limit $1""",
            limit,
        )
        return [record for row in rows if (record := await self.get(row["run_id"])) is not None]

    async def reconcile_terminal_execution(self, run_id: str) -> str:
        """Finalize a committed terminal result without executing runtime work again."""
        if self._pool is None:
            async with self._lock:
                terminals = [item for item in self._events.get(run_id, []) if item.get("terminal")]
                if len(terminals) != 1 or not terminals[0].get("result"):
                    record = self._records[run_id]
                    record.status = "failed"
                    record.error = {"code": "runtime_terminal_result_missing", "retryable": False}
                    record.owner_id = record.lease_expires_at = record.heartbeat_at = None
                    record.cancel_requested = False
                    return "quarantined"
                result = dict(terminals[0]["result"])
                status = str(result.get("status") or "")
                if status not in TERMINAL_STATUSES:
                    return "quarantined"
                record = self._records[run_id]
                record.status, record.result = status, result
                record.owner_id = record.lease_expires_at = record.heartbeat_at = None
                record.cancel_requested = False
                return "reconciled"
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                execution = await connection.fetchrow("select last_operation_id from runtime_executions where run_id=$1 for update", run_id)
                terminals = await connection.fetch("select result from runtime_events where run_id=$1 and terminal=true for update", run_id)
                if execution is None:
                    return "quarantined"
                if len(terminals) != 1 or _json_object(terminals[0]["result"]) is None:
                    invariant_error = {"code": "runtime_terminal_result_missing", "retryable": False}
                    await connection.execute(
                        """update runtime_executions set status='failed', error=$2::jsonb, cancel_requested=false,
                           owner_id=null, lease_expires_at=null, heartbeat_at=null, updated_at=now() where run_id=$1""",
                        run_id, json.dumps(invariant_error),
                    )
                    if execution["last_operation_id"]:
                        await connection.execute(
                            "update runtime_operations set status='failed' where run_id=$1 and operation_id=$2",
                            run_id, execution["last_operation_id"],
                        )
                    return "quarantined"
                result = _json_object(terminals[0]["result"])
                status = str((result or {}).get("status") or "")
                if result is None or status not in TERMINAL_STATUSES:
                    return "quarantined"
                await connection.execute(
                    """update runtime_executions set status=$2, result=$3::jsonb, cancel_requested=false,
                       owner_id=null, lease_expires_at=null, heartbeat_at=null, updated_at=now() where run_id=$1""",
                    run_id, status, json.dumps(_json_safe(result)),
                )
                if execution["last_operation_id"]:
                    await connection.execute(
                        "update runtime_operations set status=$3, result=$4::jsonb where run_id=$1 and operation_id=$2",
                        run_id, execution["last_operation_id"], status, json.dumps(_json_safe(result)),
                    )
                return "reconciled"
