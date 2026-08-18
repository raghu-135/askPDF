from __future__ import annotations

import sys
from typing import Any, Dict

from app.agent.tool_registry import (
    known_tool_contract_ids,
    tool_contracts_by_id as _default_tool_contracts_by_id,
)
from app.agent_workflows.node_catalog import node_type_allowed_tool_contract_ids
from app.agent_workflows.workflow_requirements import (
    REQUIRED_TOOL_NODE_TYPES,
    workflow_node_tool_requirements_for_allowed_tools,
)


def known_workflow_tool_ids() -> set[str]:
    return known_tool_contract_ids()


def tool_contracts_by_id() -> Dict[str, list[Dict[str, Any]]]:
    validator_module = sys.modules.get("app.agent_workflows.validator")
    accessor = getattr(validator_module, "tool_contracts_by_id", _default_tool_contracts_by_id)
    return accessor()


def collect_tool_permission_errors(spec: Dict[str, Any], allowed_tool_ids: set[str]) -> list[str]:
    errors: list[str] = []
    contracts_by_id = tool_contracts_by_id()
    workflow_id = spec.get("workflow_id")
    node_tool_requirements = workflow_node_tool_requirements_for_allowed_tools(spec, allowed_tool_ids)
    config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
    graph = config.get("graph") if isinstance(config.get("graph"), dict) else {}
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    node_types_by_id: Dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = node.get("id")
        node_type = node.get("type")
        if not isinstance(node_id, str) or not isinstance(node_type, str):
            continue
        node_types_by_id[node_id] = node_type
        candidate_tool_ids = node_type_allowed_tool_contract_ids(node_type)
        if node_type not in REQUIRED_TOOL_NODE_TYPES or not candidate_tool_ids:
            continue
        compatible_tool_ids = sorted(candidate_tool_ids & allowed_tool_ids)
        if not compatible_tool_ids:
            errors.append(f"missing required allowed_tool_ids: {', '.join(sorted(candidate_tool_ids))}")
    for caller_node, contract_id in sorted(node_tool_requirements.items()):
        if contract_id not in allowed_tool_ids:
            continue
        contracts = contracts_by_id.get(contract_id) or []
        if not contracts:
            errors.append(f"{workflow_id} required tool contract is not registered: {contract_id}")
            continue
        caller_node_type = node_types_by_id.get(caller_node)
        if not any(
            caller_node in (contract.get("allowed_caller_nodes") or [])
            or caller_node_type in (contract.get("allowed_node_types") or [])
            for contract in contracts
        ):
            errors.append(
                f"{workflow_id} tool contract {contract_id} is not allowed from node {caller_node}"
            )
    return errors
