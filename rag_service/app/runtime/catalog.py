"""Neutral projections of the existing workflow catalog."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from runtime_protocol.contracts import (
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    ContinuationBinding,
)
from runtime_protocol.events import create_runtime_event


def definition_metadata_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    runtime = spec.get("runtime") if isinstance(spec.get("runtime"), dict) else {}
    config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
    task_policy = config.get("task_policy") if isinstance(config.get("task_policy"), dict) else {}
    managed_profile = (
        spec.get("managed_profile")
        if isinstance(spec.get("managed_profile"), Mapping)
        else {}
    )
    has_managed_profile = isinstance(spec.get("managed_profile"), Mapping)
    profile_mcp = (
        managed_profile.get("mcp")
        if isinstance(managed_profile.get("mcp"), Mapping)
        else {}
    )
    effective_tools = (
        profile_mcp.get("allowed_tool_ids", [])
        if has_managed_profile
        else config.get("allowed_tool_ids", [])
    )
    profile_memory = (
        managed_profile.get("memory")
        if isinstance(managed_profile.get("memory"), Mapping)
        else {}
    )
    profile_delegation = (
        managed_profile.get("delegation")
        if isinstance(managed_profile.get("delegation"), Mapping)
        else {}
    )
    profile_skills = (
        managed_profile.get("skills")
        if isinstance(managed_profile.get("skills"), Mapping)
        else {}
    )
    effective_task_policy = (
        managed_profile.get("task_policy")
        if isinstance(managed_profile.get("task_policy"), Mapping)
        else {}
    ) if has_managed_profile else task_policy
    graph = config.get("graph") if isinstance(config.get("graph"), dict) else {}
    graph_nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    metadata = {
        "runtime_kind": runtime.get("kind"),
        "graph_node_types": sorted({
            str(node.get("type"))
            for node in graph_nodes
            if isinstance(node, dict) and node.get("type")
        }),
        "allowed_tool_ids": sorted({str(item) for item in effective_tools if item}),
        "task_profiles": sorted({str(item) for item in effective_task_policy.get("profiles", []) if item}),
        "runtime_policy": {
            "allowed_tool_ids": sorted({str(item) for item in effective_tools if item}),
            "allow_persistent_memory": bool(
                profile_memory.get("persistent", False) if has_managed_profile else config.get("allow_persistent_memory", False)
            ),
            "allow_subagents": bool(
                profile_delegation.get("enabled", False) if has_managed_profile else config.get("allow_subagents", False)
            ),
            "skills": sorted({
                str(item)
                for item in (profile_skills.get("enabled", []) if has_managed_profile else config.get("skills", []))
                if item
            }),
            "approval_enabled": bool(effective_task_policy.get("approval_enabled", True)),
            "external_context_enabled": bool(
                profile_mcp.get("runtime_profile") == "askpdf-deep-external"
                if has_managed_profile
                else config.get("use_web_search", False)
            ),
        },
    }
    return metadata


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
    payload = dict(result.output) if isinstance(result.output, Mapping) else {"answer": result.output}
    payload.update({
        "status": result.status,
        "clarification_options": clarification_options or None,
        "pending_interrupt": interruption or None,
        "agent_error": dict(result.error or {}),
        "runtime_metadata": dict(result.runtime_metadata or {}),
        "runtime_binding": result.continuation.to_dict() if result.continuation else None,
        "runtime_task_result": result.task_result.to_dict() if result.task_result else payload.get("runtime_task_result"),
        "orchestration_delta": result.orchestration_delta.to_dict() if result.orchestration_delta else None,
    })
    return payload


def event_from_source(
    event: Mapping[str, Any],
    *,
    run_id: str,
    sequence: int,
    event_id: str | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> AgentRuntimeEvent:
    data = dict(event.get("data") or {})
    kind = event.get("event") or event.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("runtime event must include a non-empty event or kind")
    return create_runtime_event(
        event_id=str(event_id or data.get("event_id") or f"{run_id}:{sequence}"),
        run_id=run_id,
        sequence=sequence,
        kind=kind,
        payload=data,
        occurred_at=data.get("occurred_at") or data.get("timestamp"),
        trace_id=data.get("trace_id"),
        source_metadata=dict(source_metadata or {"source_event": kind}),
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
