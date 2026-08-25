"""Framework-neutral continuation cleanup helpers."""

from __future__ import annotations

from typing import Any, Iterable

from app.runtime.capability_resolver import discover_adapter_capabilities
from app.runtime.catalog import continuation_from_run, definition_from_run
from app.runtime.contracts import RuntimeOperationId, RuntimeSupportLevel
from app.runtime.registry import adapter_for_definition


async def delete_run_continuation(run: Any) -> Any:
    definition = definition_from_run(run)
    if not getattr(run, "runtime_binding_json", None):
        return []
    binding = continuation_from_run(run)
    if binding is None:
        return []
    adapter = adapter_for_definition(definition)
    capabilities, _error = await discover_adapter_capabilities(adapter)
    if capabilities is None:
        return []
    descriptor = capabilities.operations.get(RuntimeOperationId.RUN_CONTINUATION_CLEANUP)
    if descriptor is None or not descriptor.enabled or descriptor.support is RuntimeSupportLevel.UNSUPPORTED:
        return []
    return await adapter.delete_continuation(binding)


async def delete_run_continuations(runs: Iterable[Any]) -> list[Any]:
    results = []
    for run in runs:
        results.append(await delete_run_continuation(run))
    return results
