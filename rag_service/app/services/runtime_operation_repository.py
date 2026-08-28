"""Durable product-side idempotency for runtime control operations."""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.db import AgentRuntimeOperation, async_session_maker
from app.time_utils import utc_now


class RuntimeOperationConflict(Exception):
    def __init__(self, code: str, message: str, *, operation: AgentRuntimeOperation | None = None):
        self.code = code
        self.operation = operation
        super().__init__(message)


def runtime_operation_lease_seconds() -> int:
    raw = os.getenv("RUNTIME_OPERATION_LEASE_SECONDS", "300")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("RUNTIME_OPERATION_LEASE_SECONDS must be an integer") from exc
    if value <= 0:
        raise RuntimeError("RUNTIME_OPERATION_LEASE_SECONDS must be greater than zero")
    return value


async def claim_runtime_operation(
    *,
    run_id: str,
    operation: str,
    idempotency_key: str,
    request_fingerprint: str,
) -> AgentRuntimeOperation:
    """Claim an operation key or return its existing durable record."""

    async with async_session_maker() as session:
        async with session.begin():
            claimed_at = utc_now()
            claim_expires_at = claimed_at + timedelta(seconds=runtime_operation_lease_seconds())
            insert_result = await session.execute(
                insert(AgentRuntimeOperation)
                .values(
                    run_id=run_id,
                    operation=operation,
                    idempotency_key=idempotency_key,
                    request_fingerprint=request_fingerprint,
                    claimed_at=claimed_at,
                    claim_expires_at=claim_expires_at,
                )
                .on_conflict_do_nothing(
                    index_elements=["run_id", "operation", "idempotency_key"],
                )
            )
            result = await session.execute(
                select(AgentRuntimeOperation)
                .where(
                    AgentRuntimeOperation.run_id == run_id,
                    AgentRuntimeOperation.operation == operation,
                    AgentRuntimeOperation.idempotency_key == idempotency_key,
                )
                .with_for_update()
            )
            existing = result.scalar_one_or_none()
            if insert_result.rowcount == 1 and existing is not None:
                return existing
            if existing is not None:
                if existing.request_fingerprint != request_fingerprint:
                    raise RuntimeOperationConflict(
                        "runtime_operation_idempotency_conflict",
                        "The idempotency key was already used with a different request",
                        operation=existing,
                    )
                existing_expiry = getattr(existing, "claim_expires_at", None)
                if existing.status == "in_progress" and (
                    existing_expiry is None or existing_expiry > claimed_at
                ):
                    raise RuntimeOperationConflict(
                        "runtime_operation_in_progress",
                        "The runtime operation is already in progress",
                        operation=existing,
                    )
                if (
                    existing.status == "failed" and bool((existing.error_json or {}).get("retryable"))
                ) or (
                    existing.status == "in_progress"
                    and existing_expiry is not None
                    and existing_expiry <= claimed_at
                ):
                    existing.status = "in_progress"
                    existing.error_json = None
                    existing.result_json = {}
                    existing.completed_at = None
                    existing.claimed_at = claimed_at
                    existing.claim_expires_at = claim_expires_at
                return existing

            raise RuntimeOperationConflict(
                "runtime_operation_claim_failed",
                "The runtime operation idempotency record could not be claimed",
            )


async def complete_runtime_operation(
    operation_id: str,
    *,
    result: Mapping[str, Any],
) -> AgentRuntimeOperation | None:
    return await _finish_runtime_operation(operation_id, status="completed", result=result, error=None)


async def fail_runtime_operation(
    operation_id: str,
    *,
    error: Mapping[str, Any],
) -> AgentRuntimeOperation | None:
    return await _finish_runtime_operation(operation_id, status="failed", result=None, error=error)


async def _finish_runtime_operation(
    operation_id: str,
    *,
    status: str,
    result: Mapping[str, Any] | None,
    error: Mapping[str, Any] | None,
) -> AgentRuntimeOperation | None:
    async with async_session_maker() as session:
        async with session.begin():
            operation_record = await session.get(AgentRuntimeOperation, operation_id, with_for_update=True)
            if operation_record is None:
                return None
            operation_record.status = status
            operation_record.result_json = dict(result or {})
            operation_record.error_json = dict(error) if error is not None else None
            operation_record.completed_at = utc_now()
            return operation_record
