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
from datetime import datetime, timezone
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


@dataclass
class ExecutionRecord:
    run_id: str
    operation: str
    request: dict[str, Any]
    payload: dict[str, Any]
    status: str = "queued"
    cancel_requested: bool = False
    next_sequence: int = 1
    continuation: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


class ExecutionStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or os.getenv("AGENT_RUNTIME_EXECUTION_DATABASE_URL", "")
        self._records: dict[str, ExecutionRecord] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._condition = None
        self._pool = None

    @property
    def durable(self) -> bool:
        return bool(self.database_url)

    async def initialize(self) -> None:
        if not self.database_url:
            return
        import asyncpg

        self._pool = await asyncpg.create_pool(self.database_url, min_size=1, max_size=5)
        async with self._pool.acquire() as connection:
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
                    continuation jsonb,
                    result jsonb,
                    error jsonb,
                    created_at timestamptz not null default now(),
                    updated_at timestamptz not null default now()
                );
                create table if not exists runtime_events (
                    run_id text not null references runtime_executions(run_id) on delete cascade,
                    sequence integer not null,
                    event_id text not null,
                    kind text not null,
                    payload jsonb not null,
                    terminal boolean not null default false,
                    result jsonb,
                    created_at timestamptz not null default now(),
                    primary key (run_id, sequence),
                    unique (run_id, event_id)
                );
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
                    # Failed/empty/cancelled executions are retryable. Clear
                    # their terminal event so a retry cannot short-circuit on
                    # the previous attempt's result.
                    self._records[run_id] = record
                    self._events[run_id] = []
                    return record
                return existing
            self._records[run_id] = record
            self._events.setdefault(run_id, [])
            return record
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                existing = await connection.fetchrow(
                    "select operation, status from runtime_executions where run_id=$1 for update",
                    run_id,
                )
                if (
                    existing is not None
                    and operation == "start"
                    and existing["status"] in {"failed", "cancelled", "no_continuation"}
                ):
                    await connection.execute("delete from runtime_events where run_id=$1", run_id)
                    await connection.execute(
                        """update runtime_executions
                           set operation=$2, request=$3::jsonb, payload=$4::jsonb,
                               status='queued', cancel_requested=false, next_sequence=1,
                               continuation=null, result=null, error=null, updated_at=now()
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
            status=row["status"], cancel_requested=row["cancel_requested"], next_sequence=row["next_sequence"],
            continuation=_json_object(row["continuation"]),
            result=_json_object(row["result"]), error=_json_object(row["error"]),
            created_at=row["created_at"].isoformat(), updated_at=row["updated_at"].isoformat(),
        )

    async def set_status(self, run_id: str, status: str, *, result: Mapping[str, Any] | None = None, error: Mapping[str, Any] | None = None) -> None:
        if self._pool is None:
            record = self._records[run_id]
            record.status, record.result, record.error, record.updated_at = status, dict(result) if result else None, dict(error) if error else None, _now()
            return
        await self._pool.execute(
            "update runtime_executions set status=$2, result=coalesce($3::jsonb,result), error=coalesce($4::jsonb,error), updated_at=now() where run_id=$1",
            run_id, status, json.dumps(_json_safe(dict(result))) if result else None, json.dumps(_json_safe(dict(error))) if error else None,
        )

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

    async def append(self, run_id: str, event: Mapping[str, Any], result: Mapping[str, Any] | None = None) -> dict[str, Any]:
        item = dict(event)
        if self._pool is None:
            record = self._records[run_id]
            existing = next((value for value in self._events.setdefault(run_id, []) if value["event_id"] == item["event_id"]), None)
            if existing is not None:
                return existing
            item["sequence"] = record.next_sequence
            record.next_sequence += 1
            item["result"] = dict(result) if result else None
            self._events[run_id].append(item)
            if item.get("continuation"):
                record.continuation = dict(item["continuation"])
            return item
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow("select next_sequence from runtime_executions where run_id=$1 for update", run_id)
                sequence = int(row["next_sequence"])
                item["sequence"] = sequence
                await connection.execute(
                    """insert into runtime_events(run_id, sequence, event_id, kind, payload, terminal, result)
                       values($1,$2,$3,$4,$5::jsonb,$6,$7::jsonb)
                       on conflict (run_id, event_id) do nothing""",
                    run_id, sequence, item["event_id"], item.get("kind", "runtime.event"), json.dumps(_json_safe(item.get("payload") or {})), bool(item.get("terminal")), json.dumps(_json_safe(dict(result))) if result else None,
                )
                await connection.execute("update runtime_executions set next_sequence=next_sequence+1, updated_at=now() where run_id=$1", run_id)
        return item

    async def events_after(self, run_id: str, sequence: int = 0) -> list[dict[str, Any]]:
        if self._pool is None:
            return [dict(item) for item in self._events.get(run_id, []) if int(item.get("sequence", 0)) > sequence]
        rows = await self._pool.fetch("select * from runtime_events where run_id=$1 and sequence>$2 order by sequence", run_id, sequence)
        events = []
        for row in rows:
            item = dict(row)
            item["payload"] = _json_object(item.get("payload")) or {}
            item["result"] = _json_object(item.get("result"))
            events.append(item)
        return events

    async def nonterminal(self) -> list[ExecutionRecord]:
        if self._pool is None:
            return [record for record in self._records.values() if record.status not in {"completed", "failed", "cancelled"}]
        rows = await self._pool.fetch("select run_id from runtime_executions where status not in ('completed','failed','cancelled','no_continuation')")
        return [record for row in rows if (record := await self.get(row["run_id"])) is not None]
