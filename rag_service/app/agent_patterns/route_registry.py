from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


ROUTE_FUNCTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    "router_route": {
        "display_name": "Router Route",
        "description": "Routes intent classification decisions from the router node.",
        "implementation_kind": "runtime_builtin",
        "runtime_supported": True,
        "builder_exposable": True,
        "allowed_source_types": ["router"],
        "route_labels": ["document", "memory", "timeline", "web", "direct", "clarify"],
    },
    "planner_route": {
        "display_name": "Planner Route",
        "description": "Routes planner decisions to execution, direct answer, or clarification paths.",
        "implementation_kind": "runtime_builtin",
        "runtime_supported": True,
        "builder_exposable": True,
        "allowed_source_types": ["planner"],
        "route_labels": ["execute", "direct", "clarify"],
    },
    "evaluator_route": {
        "display_name": "Evaluator Route",
        "description": "Routes evidence evaluator decisions to answer, replan, or budget-exhausted paths.",
        "implementation_kind": "runtime_builtin",
        "runtime_supported": True,
        "builder_exposable": True,
        "allowed_source_types": ["evidence_evaluator"],
        "route_labels": ["answer", "replan", "answer_budget_exhausted"],
    },
    "hitl_gate_route": {
        "display_name": "HITL Gate Route",
        "description": "Routes human review decisions according to the gate policy.",
        "implementation_kind": "runtime_builtin",
        "runtime_supported": True,
        "builder_exposable": True,
        "allowed_source_types": ["hitl_gate"],
        "route_labels": None,
    },
}

REQUIRED_ROUTE_FUNCTION_KEYS = {
    "display_name",
    "description",
    "implementation_kind",
    "runtime_supported",
    "builder_exposable",
    "allowed_source_types",
    "route_labels",
}

SUPPORTED_IMPLEMENTATION_KINDS = {"runtime_builtin", "metadata_only"}


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

        missing = sorted(REQUIRED_ROUTE_FUNCTION_KEYS - set(metadata))
        if missing:
            errors.append(f"{route_fn} missing registry keys: {', '.join(missing)}")

        for key in ("display_name", "description"):
            if not isinstance(metadata.get(key), str) or not metadata.get(key):
                errors.append(f"{route_fn}.{key} must be a non-empty string")

        implementation_kind = metadata.get("implementation_kind")
        if implementation_kind not in SUPPORTED_IMPLEMENTATION_KINDS:
            errors.append(
                f"{route_fn}.implementation_kind must be one of: "
                f"{', '.join(sorted(SUPPORTED_IMPLEMENTATION_KINDS))}"
            )

        for key in ("runtime_supported", "builder_exposable"):
            if not isinstance(metadata.get(key), bool):
                errors.append(f"{route_fn}.{key} must be a boolean")

        if metadata.get("builder_exposable") is True and metadata.get("runtime_supported") is not True:
            errors.append(f"{route_fn}.builder_exposable requires runtime_supported true")
        if metadata.get("runtime_supported") is True and implementation_kind != "runtime_builtin":
            errors.append(f"{route_fn}.runtime_supported requires implementation_kind runtime_builtin")

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
        elif isinstance(route_labels, list) and len(route_labels) != len(set(route_labels)):
            errors.append(f"{route_fn}.route_labels must not contain duplicates")
    return errors


def get_route_function_metadata(route_fn: str) -> Dict[str, Any]:
    return deepcopy(ROUTE_FUNCTION_REGISTRY.get(route_fn) or {})


def known_route_function_ids() -> set[str]:
    return set(ROUTE_FUNCTION_REGISTRY)


def builder_exposable_route_function_ids() -> set[str]:
    return {
        route_fn
        for route_fn, metadata in ROUTE_FUNCTION_REGISTRY.items()
        if metadata.get("builder_exposable") is True and metadata.get("runtime_supported") is True
    }


def route_function_runtime_supported(route_fn: str) -> bool:
    metadata = ROUTE_FUNCTION_REGISTRY.get(route_fn) or {}
    return (
        metadata.get("runtime_supported") is True
        and metadata.get("implementation_kind") == "runtime_builtin"
    )


def route_function_allowed_for_node_type(route_fn: str, node_type: str) -> bool:
    metadata = ROUTE_FUNCTION_REGISTRY.get(route_fn) or {}
    return node_type in set(metadata.get("allowed_source_types") or [])


def route_function_labels(route_fn: str) -> Optional[set[str]]:
    metadata = ROUTE_FUNCTION_REGISTRY.get(route_fn) or {}
    labels = metadata.get("route_labels")
    if labels is None:
        return None
    return {str(item) for item in labels}
