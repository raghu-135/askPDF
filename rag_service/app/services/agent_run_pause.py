"""Control-plane submission of cooperative runtime pause requests."""

from __future__ import annotations

from typing import Any, Mapping

from app.runtime.capability_resolver import require_capability
from app.runtime.catalog import continuation_from_run, definition_from_run
from runtime_protocol.contracts import AgentRuntimeRequest, RuntimeOperationId
from runtime_protocol.errors import RuntimeError
from app.runtime.registry import RuntimeRegistry, get_runtime_registry


def _request(task: Any, run: Any) -> AgentRuntimeRequest:
    definition = definition_from_run(run)
    return AgentRuntimeRequest(
        run_id=str(run.id),
        thread_id=str(run.thread_id),
        definition_id=definition.definition_id,
        framework=definition.framework,
        builder_id=definition.builder_id,
        task_id=str(task.id),
        continuation=continuation_from_run(run),
    )


async def request_task_pause(
    task: Any,
    run: Any | None,
    *,
    registry: RuntimeRegistry | None = None,
) -> Mapping[str, Any]:
    if run is None:
        return {"status": "not_required", "task_id": str(task.id)}
    selected = registry or get_runtime_registry()
    definition = definition_from_run(run)
    await require_capability(definition, RuntimeOperationId.TASK_PAUSE, registry=selected, run=run)
    adapter = selected.get(definition)
    # In-process LangGraph observes the control-plane task row directly. The
    # HTTP deployment needs the explicit runtime control request below.
    if not bool(getattr(adapter, "supports_external_task_pause", False)):
        return {"status": "pause_requested", "task_id": str(task.id), "run_id": str(run.id), "runtime_confirmation": "local"}
    try:
        result = await adapter.pause(_request(task, run))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError.from_exception(
            exc,
            code="runtime_transport_error",
            retryable=True,
            safe_message="Runtime pause could not be submitted",
            details={"operation_id": RuntimeOperationId.TASK_PAUSE.value, "run_id": str(run.id)},
        ) from exc
    value = dict(result) if isinstance(result, Mapping) else {"result": result}
    return {**value, "task_id": str(task.id), "run_id": str(run.id)}
