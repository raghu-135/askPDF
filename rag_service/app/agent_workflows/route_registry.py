from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from app.agent_workflows.enums import (
    EvaluatorRoute,
    PlannerRoute,
    RouteFunctionId,
    RouterRoute,
    WorkflowNodeType,
)


ROUTE_FUNCTION_REGISTRY: Dict[str, Dict[str, Any]] = {
    RouteFunctionId.ROUTER.value: {
        "allowed_source_types": [WorkflowNodeType.ROUTER.value],
        "route_labels": [route.value for route in RouterRoute],
        "target_types_by_label": {
            RouterRoute.DOCUMENT.value: [WorkflowNodeType.RETRIEVAL_WORKER.value],
            RouterRoute.MEMORY.value: [WorkflowNodeType.MEMORY_WORKER.value],
            RouterRoute.TIMELINE.value: [WorkflowNodeType.TIMELINE_WORKER.value],
            RouterRoute.WEB.value: [WorkflowNodeType.WEB_WORKER.value],
            RouterRoute.DIRECT.value: [WorkflowNodeType.DIRECT_ANSWER.value],
            RouterRoute.CLARIFY.value: [WorkflowNodeType.FINALIZER.value],
        },
    },
    RouteFunctionId.PLANNER.value: {
        "allowed_source_types": [WorkflowNodeType.PLANNER.value],
        "route_labels": [route.value for route in PlannerRoute],
        "target_types_by_label": {
            PlannerRoute.EXECUTE.value: [
                WorkflowNodeType.RETRIEVAL_WORKER.value,
                WorkflowNodeType.MEMORY_WORKER.value,
                WorkflowNodeType.TIMELINE_WORKER.value,
                WorkflowNodeType.WEB_WORKER.value,
            ],
            PlannerRoute.DIRECT.value: [
                WorkflowNodeType.DIRECT_ANSWER.value,
                WorkflowNodeType.FINALIZER.value,
            ],
            PlannerRoute.CLARIFY.value: [WorkflowNodeType.FINALIZER.value],
        },
    },
    RouteFunctionId.EVALUATOR.value: {
        "allowed_source_types": [WorkflowNodeType.EVIDENCE_EVALUATOR.value],
        "route_labels": [route.value for route in EvaluatorRoute],
        "target_types_by_label": {
            EvaluatorRoute.ANSWER.value: [WorkflowNodeType.SYNTHESIZER.value],
            EvaluatorRoute.REPLAN.value: [WorkflowNodeType.REPLANNER.value],
            EvaluatorRoute.ANSWER_BUDGET_EXHAUSTED.value: [WorkflowNodeType.SYNTHESIZER.value],
        },
    },
    RouteFunctionId.HITL_GATE.value: {
        "allowed_source_types": [WorkflowNodeType.HITL_GATE.value],
        "route_labels": None,
        "target_types_by_label": None,
    },
}

ROUTE_UI_OPTIONS: Dict[str, Dict[str, Dict[str, Any]]] = {
    RouteFunctionId.ROUTER.value: {
        RouterRoute.DOCUMENT.value: {"display_name": "Document question", "description": "Search uploaded documents.", "order": 0},
        RouterRoute.MEMORY.value: {"display_name": "Previous conversation", "description": "Search conversation memory.", "order": 1},
        RouterRoute.TIMELINE.value: {"display_name": "Timeline question", "description": "Search chronological thread events.", "order": 2},
        RouterRoute.WEB.value: {"display_name": "Current information", "description": "Search approved external sources.", "order": 3},
        RouterRoute.DIRECT.value: {"display_name": "Answer directly", "description": "Answer without retrieval.", "order": 4},
        RouterRoute.CLARIFY.value: {"display_name": "Needs clarification", "description": "Ask the user for more detail.", "order": 5},
    },
    RouteFunctionId.PLANNER.value: {
        PlannerRoute.EXECUTE.value: {"display_name": "Run the plan", "description": "Continue through planned retrieval.", "order": 0},
        PlannerRoute.DIRECT.value: {"display_name": "Answer directly", "description": "No retrieval steps are needed.", "order": 1},
        PlannerRoute.CLARIFY.value: {"display_name": "Needs clarification", "description": "Ask for missing information.", "order": 2},
    },
    RouteFunctionId.EVALUATOR.value: {
        EvaluatorRoute.ANSWER.value: {"display_name": "Evidence is sufficient", "description": "Continue to synthesis.", "order": 0},
        EvaluatorRoute.REPLAN.value: {"display_name": "Search again", "description": "Evidence gaps require another bounded search.", "order": 1},
        EvaluatorRoute.ANSWER_BUDGET_EXHAUSTED.value: {"display_name": "Answer with available evidence", "description": "The replan budget is exhausted.", "order": 2},
    },
}


def get_route_function_registry() -> Dict[str, Dict[str, Any]]:
    registry = deepcopy(ROUTE_FUNCTION_REGISTRY)
    for route_fn, metadata in registry.items():
        metadata["route_options"] = deepcopy(ROUTE_UI_OPTIONS.get(route_fn, {}))
    return registry


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

        missing = sorted({"allowed_source_types", "route_labels", "target_types_by_label"} - set(metadata))
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

        target_types_by_label = metadata.get("target_types_by_label")
        if route_labels is None:
            if target_types_by_label is not None:
                errors.append(f"{route_fn}.target_types_by_label must be null when route_labels is null")
        elif not isinstance(target_types_by_label, dict):
            errors.append(f"{route_fn}.target_types_by_label must be an object")
        else:
            missing_targets = sorted(set(route_labels) - set(target_types_by_label))
            unknown_targets = sorted(set(target_types_by_label) - set(route_labels))
            if missing_targets:
                errors.append(
                    f"{route_fn}.target_types_by_label is missing route labels: {', '.join(missing_targets)}"
                )
            if unknown_targets:
                errors.append(
                    f"{route_fn}.target_types_by_label has unknown route labels: {', '.join(unknown_targets)}"
                )
            for label, target_types in sorted(target_types_by_label.items()):
                if not isinstance(target_types, list) or not all(
                    isinstance(item, str) and item for item in target_types
                ):
                    errors.append(
                        f"{route_fn}.target_types_by_label.{label} must be a list of non-empty strings"
                    )
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
