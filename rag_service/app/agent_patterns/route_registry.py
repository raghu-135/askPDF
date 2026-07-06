from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


ROUTE_FUNCTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "router_route": {
        "allowed_source_types": ["router"],
        "route_labels": ["document", "memory", "timeline", "web", "direct", "clarify"],
    },
    "planner_route": {
        "allowed_source_types": ["planner"],
        "route_labels": ["execute", "direct", "clarify"],
    },
    "evaluator_route": {
        "allowed_source_types": ["evidence_evaluator"],
        "route_labels": ["answer", "replan", "answer_budget_exhausted"],
    },
    "hitl_gate_route": {
        "allowed_source_types": ["hitl_gate"],
        "route_labels": None,
    },
}


def get_route_function_registry() -> Dict[str, Dict[str, Any]]:
    return deepcopy(ROUTE_FUNCTION_REGISTRY)


def get_route_function_metadata(route_fn: str) -> Dict[str, Any]:
    return deepcopy(ROUTE_FUNCTION_REGISTRY.get(route_fn) or {})


def known_route_function_ids() -> set[str]:
    return set(ROUTE_FUNCTION_REGISTRY)


def route_function_allowed_for_node_type(route_fn: str, node_type: str) -> bool:
    metadata = ROUTE_FUNCTION_REGISTRY.get(route_fn) or {}
    return node_type in set(metadata.get("allowed_source_types") or [])


def route_function_labels(route_fn: str) -> Optional[set[str]]:
    metadata = ROUTE_FUNCTION_REGISTRY.get(route_fn) or {}
    labels = metadata.get("route_labels")
    if labels is None:
        return None
    return {str(item) for item in labels}
