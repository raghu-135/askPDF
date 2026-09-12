"""Framework-neutral continuation cleanup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.runtime.catalog import definition_from_run
from app.runtime.capability_resolver import resolve_definition_capability_resolution
from app.runtime.registry import RuntimeRegistry, get_runtime_registry
from runtime_protocol.contracts import (
    RuntimeCleanupResult, RuntimeCleanupStatus, RuntimeOperationId, RuntimeSupportLevel,
)


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
        # Explicitly unsupported cleanup is an intentional ownership policy;
        # failed discovery or disabled supported cleanup must never imply it.
        return self.status in {"cleaned", "already_cleaned", "not_bound", "unsupported"}


async def cleanup_run(
    run: Any,
    *,
    registry: RuntimeRegistry | None = None,
) -> RunCleanupOutcome:
    run_id = str(getattr(run, "id", ""))
    try:
        definition = definition_from_run(run)
        registry = registry or get_runtime_registry()
        adapter = registry.get(definition)
        # Cleanup also applies to completed/unbound runs. Do not apply the
        # run-level checkpoint/resume eligibility projection here.
        resolution = await resolve_definition_capability_resolution(
            definition, registry=registry, adapter=adapter,
        )
        if resolution.error or not resolution.runtime_available:
            return RunCleanupOutcome(run_id, "failed", error=resolution.error or "runtime unavailable")
        descriptor = resolution.capabilities.operations.get(RuntimeOperationId.RUN_CLEANUP)
        if descriptor is None:
            return RunCleanupOutcome(run_id, "failed", error="runtime cleanup capability missing")
        if descriptor.support == RuntimeSupportLevel.UNSUPPORTED:
            return RunCleanupOutcome(run_id, "unsupported")
        if not descriptor.enabled:
            return RunCleanupOutcome(run_id, "failed", error=descriptor.disabled_reason or "runtime cleanup disabled")
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
