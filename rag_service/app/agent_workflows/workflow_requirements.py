from __future__ import annotations

from typing import Any, Dict

from app.agent_workflows.node_catalog import node_type_allowed_tool_contract_ids


REQUIRED_TOOL_NODE_TYPES = {"retrieval_worker", "memory_worker", "timeline_worker", "web_worker"}


def workflow_required_tool_ids(spec: Dict[str, Any]) -> set[str]:
    config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
    allowed_tool_ids = config.get("allowed_tool_ids") if isinstance(config.get("allowed_tool_ids"), list) else []
    return {str(tool_id) for tool_id in allowed_tool_ids if isinstance(tool_id, str) and tool_id}


def workflow_node_tool_requirements(spec: Dict[str, Any]) -> Dict[str, str]:
    required_tool_ids = workflow_required_tool_ids(spec)
    return workflow_node_tool_requirements_for_allowed_tools(spec, required_tool_ids)


def workflow_required_tool_ids_from_nodes(spec: Dict[str, Any]) -> set[str]:
    config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
    graph = config.get("graph") if isinstance(config.get("graph"), dict) else {}
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    required_tool_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = node.get("type")
        if isinstance(node_type, str) and node_type in REQUIRED_TOOL_NODE_TYPES:
            required_tool_ids.update(node_type_allowed_tool_contract_ids(node_type))
    return required_tool_ids


def workflow_node_tool_requirements_for_allowed_tools(spec: Dict[str, Any], allowed_tool_ids: set[str]) -> Dict[str, str]:
    config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
    graph = config.get("graph") if isinstance(config.get("graph"), dict) else {}
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    requirements: Dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        node_type = node.get("type")
        if not isinstance(node_id, str) or not isinstance(node_type, str):
            continue
        compatible_tool_ids = sorted(node_type_allowed_tool_contract_ids(node_type) & allowed_tool_ids)
        if len(compatible_tool_ids) == 1:
            requirements[node_id] = compatible_tool_ids[0]
    return requirements
