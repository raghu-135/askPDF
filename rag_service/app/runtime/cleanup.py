"""Framework-neutral continuation cleanup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.runtime.catalog import definition_from_run
from app.runtime.registry import RuntimeRegistry, get_runtime_registry
from runtime_protocol.contracts import RuntimeCleanupResult, RuntimeCleanupStatus


@dataclass(frozen=True)
class RunCleanupOutcome:
    run_id: str
    status: str
    adapter_result: Any = None
    error: Any = None

    @property
    def cleaned(self) -> bool:
        return self.status in {"cleaned", "already_cleaned", "not_bound"}

    @property
    def owner_deletion_allowed(self) -> bool:
        # Hermes owns native execution/session cleanup and deliberately does
        # not participate in the LangGraph run-cleanup contract.  This is a
        # framework boundary, not a LangGraph cleanup fallback.
        return self.status in {"cleaned", "already_cleaned", "not_bound", "unsupported"}


async def cleanup_run(
    run: Any,
    *,
    registry: RuntimeRegistry | None = None,
) -> RunCleanupOutcome:
    run_id = str(getattr(run, "id", ""))
    definition = definition_from_run(run)
    if str(getattr(definition, "framework", "")) != "langgraph":
        return RunCleanupOutcome(run_id=run_id, status="unsupported")
    registry = registry or get_runtime_registry()
    adapter = registry.get(definition)
    try:
        result = await adapter.cleanup_run(run_id)
    except Exception as exc:
        return RunCleanupOutcome(run_id=run_id, status="failed", error=str(exc))
    if not isinstance(result, RuntimeCleanupResult):
        return RunCleanupOutcome(
            run_id=run_id,
            status="failed",
            error="runtime cleanup response must be RuntimeCleanupResult",
            adapter_result=result,
        )
    if result.run_id != run_id:
        return RunCleanupOutcome(
            run_id=run_id,
            status="failed",
            error="runtime cleanup response returned a different run_id",
            adapter_result=result,
        )
    result_status = result.status.value if isinstance(result.status, RuntimeCleanupStatus) else str(result.status)
    return RunCleanupOutcome(run_id=run_id, status=result_status, adapter_result=result)


async def cleanup_runs(
    runs: Iterable[Any],
    *,
    registry: RuntimeRegistry | None = None,
) -> list[RunCleanupOutcome]:
    results: list[RunCleanupOutcome] = []
    for run in runs:
        results.append(await cleanup_run(run, registry=registry))
    return results
