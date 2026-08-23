"""Neutral projections of the existing workflow catalog."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
)
from app.runtime.events import create_runtime_event


def definition_metadata_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    runtime = spec.get("runtime") if isinstance(spec.get("runtime"), dict) else {}
    config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
    graph = config.get("graph") if isinstance(config.get("graph"), dict) else {}
    graph_nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    task_policy = config.get("task_policy") if isinstance(config.get("task_policy"), dict) else {}
    return {
        "runtime_kind": runtime.get("kind"),
        "graph_node_types": sorted({
            str(node.get("type"))
            for node in graph_nodes
            if isinstance(node, dict) and node.get("type")
        }),
        "allowed_tool_ids": sorted({str(item) for item in config.get("allowed_tool_ids", []) if item}),
        "task_profiles": sorted({str(item) for item in task_policy.get("profiles", []) if item}),
    }


def definition_from_workflow(workflow: Any) -> AgentDefinition:
    metadata = workflow.metadata_json if isinstance(getattr(workflow, "metadata_json", None), dict) else {}
    spec = workflow.spec_json if isinstance(getattr(workflow, "spec_json", None), dict) else {}
    runtime = spec.get("runtime") if isinstance(spec.get("runtime"), dict) else {}
    features = runtime.get("features") if isinstance(runtime.get("features"), dict) else {}
    framework = str(getattr(workflow, "framework", None) or metadata.get("framework") or "").strip()
    builder_id = str(getattr(workflow, "builder_id", None) or metadata.get("builder_id") or "").strip()
    if not framework or not builder_id:
        raise ValueError(f"Workflow {workflow.id!r} has no concrete runtime identity")
    category = getattr(workflow, "category", None) or metadata.get("category")
    return AgentDefinition(
        definition_id=str(workflow.id),
        framework=framework,
        builder_id=builder_id,
        category=str(category) if category else None,
        display_name=str(getattr(workflow, "name", None) or workflow.id),
        capabilities=dict(features),
        definition_metadata=definition_metadata_from_spec(spec),
    )


def definition_from_run(run: Any) -> AgentDefinition:
    """Build the runtime definition from one frozen persisted run."""

    framework = str(getattr(run, "framework", None) or "").strip()
    builder_id = str(getattr(run, "builder_id", None) or "").strip()
    definition_id = str(getattr(run, "workflow_id", None) or "").strip()
    if not definition_id or not framework or not builder_id:
        raise ValueError("Persisted agent run is missing concrete runtime identity")
    spec = getattr(run, "resolved_spec_json", None)
    spec = spec if isinstance(spec, Mapping) else {}
    runtime = spec.get("runtime") if isinstance(spec.get("runtime"), Mapping) else {}
    features = runtime.get("features") if isinstance(runtime.get("features"), Mapping) else {}
    return AgentDefinition(
        definition_id=definition_id,
        framework=framework,
        builder_id=builder_id,
        category=getattr(run, "definition_category", None),
        capabilities=dict(features),
        definition_metadata=definition_metadata_from_spec(spec),
    )


def continuation_from_run(run: Any) -> ContinuationBinding | None:
    payload = getattr(run, "runtime_binding_json", None)
    if isinstance(payload, Mapping) and payload:
        binding_type = str(payload.get("binding_type") or "").strip()
        if not binding_type:
            raise ValueError(f"Run {run.id!r} has an invalid runtime binding")
        return ContinuationBinding(binding_type=binding_type, payload=dict(payload.get("payload") or {}))
    checkpoint_id = getattr(run, "checkpoint_thread_id", None)
    if checkpoint_id:
        return ContinuationBinding(
            binding_type="langgraph_checkpoint",
            payload={"checkpoint_thread_id": str(checkpoint_id)},
        )
    return None


def request_from_run(
    run: Any,
    *,
    input: Mapping[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    trace_id: str | None = None,
) -> AgentRuntimeRequest:
    definition = definition_from_run(run)
    return AgentRuntimeRequest(
        run_id=str(run.id),
        thread_id=str(run.thread_id),
        definition_id=definition.definition_id,
        framework=definition.framework,
        builder_id=definition.builder_id,
        input=dict(input or {}),
        options=dict(options or {}),
        task_id=getattr(run, "task_id", None),
        parent_run_id=getattr(run, "parent_run_id", None),
        continuation=continuation_from_run(run),
        trace_id=trace_id,
    )


def result_to_product_payload(result: AgentRuntimeResult) -> dict[str, Any]:
    """Project a typed runtime result into the existing product response shape."""

    clarification_options = list((result.clarification or {}).get("options") or [])
    interruption = dict(result.interruption or {})
    return {
        "status": result.status,
        "answer": result.output,
        "clarification_options": clarification_options or None,
        "pending_interrupt": interruption or None,
        "agent_error": dict(result.error or {}),
        "runtime_metadata": dict(result.runtime_metadata or {}),
        "runtime_binding": result.continuation.to_dict() if result.continuation else None,
    }


def event_from_source(
    event: Mapping[str, Any],
    *,
    run_id: str,
    sequence: int,
    event_id: str | None = None,
) -> AgentRuntimeEvent:
    data = dict(event.get("data") or {})
    kind = str(event.get("event") or event.get("kind") or "runtime.event")
    return create_runtime_event(
        event_id=str(event_id or data.get("event_id") or f"{run_id}:{sequence}"),
        run_id=run_id,
        sequence=sequence,
        kind=kind,
        payload=data,
        occurred_at=data.get("occurred_at") or data.get("timestamp"),
        trace_id=data.get("trace_id"),
        source_metadata={"framework": "langgraph", "source_event": kind}
        if kind not in {"run.completed", "run.failed", "run.cancelled"}
        else {"framework": "langgraph"},
    )


def catalog_payload(workflow: Any) -> dict[str, Any]:
    """Add neutral identity to the existing workflow API representation."""

    definition = definition_from_workflow(workflow)
    return {
        "definition_id": definition.definition_id,
        "framework": definition.framework,
        "builder_id": definition.builder_id,
        "category": definition.category,
    }
