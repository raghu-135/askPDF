"""Neutral projections of the existing workflow catalog."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from app.runtime.contracts import AgentDefinition


DEFAULT_FRAMEWORK = "langgraph"
DEFAULT_BUILDER_ID = "langgraph_graph"


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
    framework = str(getattr(workflow, "framework", None) or metadata.get("framework") or DEFAULT_FRAMEWORK)
    builder_id = str(getattr(workflow, "builder_id", None) or metadata.get("builder_id") or DEFAULT_BUILDER_ID)
    category = getattr(workflow, "category", None) or metadata.get("category")
    version = getattr(workflow, "version", None)
    return AgentDefinition(
        definition_id=str(workflow.id),
        framework=framework,
        builder_id=builder_id,
        category=str(category) if category else None,
        display_name=str(getattr(workflow, "name", None) or workflow.id),
        capabilities=dict(features),
        definition_metadata=definition_metadata_from_spec(spec),
        definition_version=str(version) if version is not None else None,
        runtime_version=str(runtime.get("version")) if runtime.get("version") else None,
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
