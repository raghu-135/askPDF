"""Framework-neutral continuation cleanup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from app.runtime.capability_resolver import resolve_run_capability_resolution
from app.runtime.catalog import continuation_from_run, definition_from_run
from app.runtime.contracts import RuntimeOperationId, RuntimeSupportLevel
from app.runtime.registry import RuntimeRegistry, get_runtime_registry


@dataclass(frozen=True)
class ContinuationCleanupOutcome:
    run_id: str
    status: str
    adapter_result: Any = None
    error: Any = None

    @property
    def cleaned(self) -> bool:
        return self.status == "cleaned"

    @property
    def owner_deletion_allowed(self) -> bool:
        return self.status in {"cleaned", "not_bound", "unsupported"}


async def delete_run_continuation(
    run: Any,
    *,
    registry: RuntimeRegistry | None = None,
) -> ContinuationCleanupOutcome:
    run_id = str(getattr(run, "id", ""))
    definition = definition_from_run(run)
    if not getattr(run, "runtime_binding_json", None):
        return ContinuationCleanupOutcome(run_id=run_id, status="not_bound")
    binding = continuation_from_run(run)
    if binding is None:
        return ContinuationCleanupOutcome(
            run_id=run_id,
            status="invalid_binding",
            error="The persisted runtime binding is invalid.",
        )
    registry = registry or get_runtime_registry()
    adapter = registry.get(definition)
    resolution = await resolve_run_capability_resolution(
        definition,
        registry=registry,
        run=run,
        adapter=adapter,
    )
    if not resolution.runtime_available:
        return ContinuationCleanupOutcome(
            run_id=run_id,
            status="unavailable",
            error=resolution.error,
        )
    capabilities = resolution.capabilities
    descriptor = capabilities.operations.get(RuntimeOperationId.RUN_CONTINUATION_CLEANUP)
    if descriptor is None or not descriptor.enabled or descriptor.support is RuntimeSupportLevel.UNSUPPORTED:
        return ContinuationCleanupOutcome(run_id=run_id, status="unsupported")
    try:
        result = await adapter.delete_continuation(binding)
    except Exception as exc:
        return ContinuationCleanupOutcome(run_id=run_id, status="failed", error=str(exc))
    return ContinuationCleanupOutcome(
        run_id=run_id,
        status="cleaned",
        adapter_result=result,
    )


async def delete_run_continuations(
    runs: Iterable[Any],
    *,
    registry: RuntimeRegistry | None = None,
) -> list[ContinuationCleanupOutcome]:
    results: list[ContinuationCleanupOutcome] = []
    for run in runs:
        results.append(await delete_run_continuation(run, registry=registry))
    return results
