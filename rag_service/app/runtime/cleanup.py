"""Framework-neutral continuation cleanup helpers."""

from __future__ import annotations

from typing import Any, Iterable

from app.runtime.catalog import continuation_from_run, definition_from_run
from app.runtime.registry import adapter_for_definition


async def delete_run_continuation(run: Any) -> Any:
    definition = definition_from_run(run)
    if definition.framework != "langgraph" and not getattr(run, "runtime_binding_json", None):
        return []
    binding = continuation_from_run(run)
    if binding is None:
        return []
    return await adapter_for_definition(definition).delete_continuation(binding)


async def delete_run_continuations(runs: Iterable[Any]) -> list[Any]:
    results = []
    for run in runs:
        results.append(await delete_run_continuation(run))
    return results
