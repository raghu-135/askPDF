"""Durable product-side idempotency for runtime control operations."""

from __future__ import annotations

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
            insert_result = await session.execute(
                insert(AgentRuntimeOperation)
                .values(
                    run_id=run_id,
                    operation=operation,
                    idempotency_key=idempotency_key,
                    request_fingerprint=request_fingerprint,
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
                if existing.status == "in_progress":
                    raise RuntimeOperationConflict(
                        "runtime_operation_in_progress",
                        "The runtime operation is already in progress",
                        operation=existing,
                    )
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
