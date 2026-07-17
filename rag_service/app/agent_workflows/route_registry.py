from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from app.agent_workflows.enums import (
    EvaluatorRoute,
    PlannerRoute,
    RouteFunctionId,
    RouterRoute,
)


ROUTE_FUNCTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    RouteFunctionId.ROUTER.value: {
        "allowed_source_types": ["router"],
        "route_labels": [route.value for route in RouterRoute],
    },
    RouteFunctionId.PLANNER.value: {
        "allowed_source_types": ["planner"],
        "route_labels": [route.value for route in PlannerRoute],
    },
    RouteFunctionId.EVALUATOR.value: {
        "allowed_source_types": ["evidence_evaluator"],
        "route_labels": [route.value for route in EvaluatorRoute],
    },
    RouteFunctionId.HITL_GATE.value: {
        "allowed_source_types": ["hitl_gate"],
        "route_labels": None,
    },
}


def get_route_function_registry() -> Dict[str, Dict[str, Any]]:
    return deepcopy(ROUTE_FUNCTION_REGISTRY)


def collect_route_function_registry_errors(registry: Dict[str, Dict[str, Any]] | None = None) -> list[str]:
    errors: list[str] = []
    source = registry if isinstance(registry, dict) else ROUTE_FUNCTION_REGISTRY
    for route_fn, metadata in sorted(source.items()):
        if not isinstance(route_fn, str) or not route_fn:
            errors.append("route function ids must be non-empty strings")
            continue
        if not isinstance(metadata, dict):
            errors.append(f"{route_fn} metadata must be an object")
            continue

        missing = sorted({"allowed_source_types", "route_labels"} - set(metadata))
        if missing:
            errors.append(f"{route_fn} missing registry keys: {', '.join(missing)}")

        allowed_source_types = metadata.get("allowed_source_types")
        if not isinstance(allowed_source_types, list) or not all(
            isinstance(item, str) and item for item in allowed_source_types
        ):
            errors.append(f"{route_fn}.allowed_source_types must be a list of non-empty strings")

        route_labels = metadata.get("route_labels")
        if route_labels is not None and (
            not isinstance(route_labels, list) or not all(isinstance(item, str) and item for item in route_labels)
        ):
            errors.append(f"{route_fn}.route_labels must be null or a list of non-empty strings")
    return errors


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
