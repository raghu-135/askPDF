"""Framework-neutral continuation cleanup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.runtime.catalog import definition_from_run
from app.runtime.registry import RuntimeRegistry, get_runtime_registry


@dataclass(frozen=True)
class ContinuationCleanupOutcome:
    run_id: str
    status: str
    adapter_result: Any = None
    error: Any = None

    @property
    def cleaned(self) -> bool:
        return self.status in {"cleaned", "already_cleaned", "not_bound"}

    @property
    def owner_deletion_allowed(self) -> bool:
        return self.status in {"cleaned", "already_cleaned", "not_bound", "unsupported"}


async def delete_run_continuation(
    run: Any,
    *,
    registry: RuntimeRegistry | None = None,
) -> ContinuationCleanupOutcome:
    run_id = str(getattr(run, "id", ""))
    definition = definition_from_run(run)
    if str(getattr(definition, "framework", "")) != "langgraph":
        return ContinuationCleanupOutcome(run_id=run_id, status="unsupported")
    registry = registry or get_runtime_registry()
    adapter = registry.get(definition)
    try:
        result = await adapter.cleanup_run(run_id)
    except Exception as exc:
        return ContinuationCleanupOutcome(run_id=run_id, status="failed", error=str(exc))
    if not isinstance(result, dict):
        return ContinuationCleanupOutcome(run_id=run_id, status="failed", error="runtime cleanup response must be an object")
    result_status = str(result.get("status") or "")
    if result_status not in {"cleaned", "already_cleaned", "not_bound"}:
        return ContinuationCleanupOutcome(run_id=run_id, status="failed", error="runtime cleanup response has an unexpected status", adapter_result=result)
    return ContinuationCleanupOutcome(run_id=run_id, status=result_status, adapter_result=result)


async def delete_run_continuations(
    runs: Iterable[Any],
    *,
    registry: RuntimeRegistry | None = None,
) -> list[ContinuationCleanupOutcome]:
    results: list[ContinuationCleanupOutcome] = []
    for run in runs:
        results.append(await delete_run_continuation(run, registry=registry))
    return results
