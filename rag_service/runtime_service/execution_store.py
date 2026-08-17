"""Durable runtime execution and event storage.

The runtime keeps execution state here rather than in an SSE generator.  A
small in-memory implementation is retained for unit tests; production uses
PostgreSQL when AGENT_RUNTIME_EXECUTION_DATABASE_URL is configured.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping


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


class LeaseLostError(RuntimeError):
    """Raised when a worker attempts to mutate a run it no longer owns."""


class ExecutionStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("AGENT_RUNTIME_EXECUTION_DATABASE_URL", "")
        self._records: dict[str, ExecutionRecord] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._condition = None
        self._pool = None
        self.owner_id = os.getenv("AGENT_RUNTIME_WORKER_ID") or f"runtime-{uuid.uuid4().hex}"
        self.lease_seconds = max(5, int(os.getenv("AGENT_RUNTIME_LEASE_SECONDS", "60")))

    @property
    def durable(self) -> bool:
        return bool(self.database_url)

    async def initialize(self) -> None:
        if not self.database_url:
            return
        import asyncpg

        self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        async with self._pool.acquire() as connection:
            auto_create = os.getenv("AGENT_RUNTIME_SCHEMA_AUTO_CREATE", "false").strip().lower() in {"1", "true", "yes", "on"}
            if not auto_create:
                await connection.execute("select 1 from runtime_executions limit 0")
                return
            await connection.execute(
                """
                create table if not exists runtime_executions (
                    run_id text primary key,
                    operation text not null,
                    request jsonb not null,
                    payload jsonb not null,
                    status text not null,
                    cancel_requested boolean not null default false,
                    next_sequence integer not null default 1,
                    attempt integer not null default 1,
                    continuation jsonb,
                    result jsonb,
                    error jsonb,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                    ,owner_id text
                    ,lease_expires_at timestamptz
                    ,heartbeat_at timestamptz
                    ,fencing_token bigint not null default 0
                );
                create table if not exists runtime_events (
                    run_id text not null references runtime_executions(run_id) on delete cascade,
                    sequence integer not null,
                    attempt integer not null default 1,
                    event_id text not null,
                    kind text not null,
                    payload jsonb not null,
                    occurred_at text,
                    trace_id text,
                    runtime_version text,
                    contract_version integer,
                    continuation jsonb,
                    terminal boolean not null default false,
                    result jsonb,
                    created_at timestamptz not null default now(),
                    primary key (run_id, sequence),
                    unique (run_id, event_id)
                );
                alter table runtime_events add column if not exists occurred_at text;
                alter table runtime_events add column if not exists trace_id text;
                alter table runtime_events add column if not exists runtime_version text;
                alter table runtime_events add column if not exists contract_version integer;
                alter table runtime_events add column if not exists continuation jsonb;
                alter table runtime_executions add column if not exists attempt integer not null default 1;
                alter table runtime_events add column if not exists attempt integer not null default 1;
                alter table runtime_executions add column if not exists owner_id text;
                alter table runtime_executions add column if not exists lease_expires_at timestamptz;
                alter table runtime_executions add column if not exists heartbeat_at timestamptz;
                alter table runtime_executions add column if not exists fencing_token bigint not null default 0;
                """
            )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def create(self, run_id: str, operation: str, request: Mapping[str, Any], payload: Mapping[str, Any]) -> ExecutionRecord:
        record = ExecutionRecord(run_id, operation, dict(request), dict(payload))
        if self._pool is None:
            if run_id in self._records:
                existing = self._records[run_id]
                if operation == "start" and existing.status in {"failed", "cancelled", "no_continuation"}:
                    # Failed/empty/cancelled executions are retryable. Keep
                    # the prior attempt for audit/replay, but make the new
                    # attempt the current durable execution.
                    existing.operation = operation
                    existing.request = dict(request)
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
                    return existing
                if (
                    operation in {"resume", "continue_run"}
                    and existing.status not in {"queued", "running"}
                    and existing.continuation is not None
                ):
                    existing.operation = operation
                    existing.request = dict(request)
                    existing.payload = dict(payload)
                    existing.status = "queued"
                    existing.cancel_requested = False
                    existing.attempt += 1
                    existing.result = None
                    existing.error = None
                    existing.owner_id = None
                    existing.lease_expires_at = None
                    existing.heartbeat_at = None
                    return existing
                return existing
            self._records[run_id] = record
            self._events.setdefault(run_id, [])
            return record
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                existing = await connection.fetchrow(
                    "select operation, status, attempt, continuation from runtime_executions where run_id=$1 for update",
                    run_id,
                )
                if (
                    existing is not None
                    and operation == "start"
                    and existing["status"] in {"failed", "cancelled", "no_continuation"}
                ):
                    await connection.execute(
                        """update runtime_executions
                           set operation=$2, request=$3::jsonb, payload=$4::jsonb,
                               status='queued', cancel_requested=false, attempt=attempt+1,
                               continuation=null, result=null, error=null, owner_id=null,
                               lease_expires_at=null, heartbeat_at=null, updated_at=now()
                           where run_id=$1""",
                        run_id, operation, json.dumps(_json_safe(dict(request))), json.dumps(_json_safe(dict(payload))),
                    )
                elif (
                    existing is not None
                    and operation in {"resume", "continue_run"}
                    and existing["status"] not in {"queued", "running"}
                    and existing["continuation"] is not None
                ):
                    await connection.execute(
                        """update runtime_executions
                           set operation=$2, request=$3::jsonb, payload=$4::jsonb,
                               status='queued', cancel_requested=false, attempt=attempt+1,
                               result=null, error=null, owner_id=null,
                               lease_expires_at=null, heartbeat_at=null, updated_at=now()
                           where run_id=$1""",
                        run_id, operation, json.dumps(_json_safe(dict(request))), json.dumps(_json_safe(dict(payload))),
                    )
                elif existing is None:
                    await connection.execute(
                        """insert into runtime_executions(run_id, operation, request, payload, status)
                           values($1,$2,$3::jsonb,$4::jsonb,'queued')""",
                        run_id, operation, json.dumps(_json_safe(dict(request))), json.dumps(_json_safe(dict(payload))),
                    )
        return await self.get(run_id)  # type: ignore[return-value]

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
        )

    async def claim(self, run_id: str, *, owner_id: str | None = None, lease_seconds: int | None = None) -> int | None:
        owner_id = owner_id or self.owner_id
        lease_seconds = lease_seconds or self.lease_seconds
        if self._pool is None:
            record = self._records.get(run_id)
            if record is None or record.status in {"completed", "failed", "cancelled", "no_continuation"}:
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
                     and status not in ('completed','failed','cancelled','no_continuation')
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
               where run_id=$1 and owner_id=$2 and fencing_token=$3 and status not in ('completed','failed','cancelled','no_continuation')""",
            run_id, owner_id, fencing_token, lease_seconds,
        )
        return result.endswith("1")

    async def set_status(self, run_id: str, status: str, *, result: Mapping[str, Any] | None = None, error: Mapping[str, Any] | None = None, owner_id: str | None = None, fencing_token: int | None = None) -> None:
        if self._pool is None:
            record = self._records[run_id]
            if owner_id is not None and (record.owner_id != owner_id or record.fencing_token != fencing_token):
                raise LeaseLostError(f"lost runtime lease for {run_id}")
            record.status, record.result, record.error, record.updated_at = status, dict(result) if result else None, dict(error) if error else None, _now()
            return
        if owner_id is None or fencing_token is None:
            raise LeaseLostError(f"runtime status mutation requires a lease for {run_id}")
        result_status = await self._pool.execute(
            """update runtime_executions set status=$2, result=coalesce($3::jsonb,result), error=coalesce($4::jsonb,error), updated_at=now()
               where run_id=$1 and owner_id=$5 and fencing_token=$6 and (lease_expires_at is null or lease_expires_at > now())""",
            run_id, status, json.dumps(_json_safe(dict(result))) if result else None, json.dumps(_json_safe(dict(error))) if error else None, owner_id, fencing_token,
        )
        if not result_status.endswith("1"):
            raise LeaseLostError(f"lost runtime lease for {run_id}")

    async def request_cancel(self, run_id: str) -> bool:
        if self._pool is None:
            record = self._records.get(run_id)
            if record is None:
                return False
            record.cancel_requested = True
            return True
        result = await self._pool.execute("update runtime_executions set cancel_requested=true, updated_at=now() where run_id=$1", run_id)
        return result.endswith("1")

    async def is_cancel_requested(self, run_id: str) -> bool:
        record = await self.get(run_id)
        return bool(record and record.cancel_requested)

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
                           trace_id, runtime_version, contract_version,
                           continuation, terminal, result
                       ) values($1,$2,$3,$4,$5,$6::jsonb,$7,$8,$9,$10,$11::jsonb,$12,$13::jsonb)
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
                    item.get("runtime_version"),
                    item.get("contract_version"),
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
        if self._pool is None:
            return [record for record in self._records.values() if record.status not in {"completed", "failed", "cancelled"}]
        rows = await self._pool.fetch("select run_id from runtime_executions where status not in ('completed','failed','cancelled','no_continuation')")
        return [record for row in rows if (record := await self.get(row["run_id"])) is not None]
