from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

from app.agent_workflows.enums import (
    AnswerQualityRoute,
    CorrectiveRetrievalRoute,
    EvaluatorRoute,
    GroundedAnswerRoute,
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
            RouterRoute.DOCUMENT.value: [WorkflowNodeType.RETRIEVAL_WORKER.value, WorkflowNodeType.SERIAL_DISPATCH.value],
            RouterRoute.THREAD_CONVERSATION_HISTORY.value: [WorkflowNodeType.THREAD_CONVERSATION_HISTORY_WORKER.value, WorkflowNodeType.SERIAL_DISPATCH.value],
            RouterRoute.DURABLE_MEMORY.value: [WorkflowNodeType.DURABLE_MEMORY_WORKER.value, WorkflowNodeType.SERIAL_DISPATCH.value],
            RouterRoute.THREAD_EVENTS.value: [WorkflowNodeType.THREAD_EVENTS_WORKER.value, WorkflowNodeType.SERIAL_DISPATCH.value],
            RouterRoute.WEB.value: [WorkflowNodeType.WEB_WORKER.value, WorkflowNodeType.SERIAL_DISPATCH.value],
            RouterRoute.COMPOUND.value: [WorkflowNodeType.PLANNER.value],
            RouterRoute.DIRECT.value: [WorkflowNodeType.DIRECT_ANSWER.value],
            RouterRoute.CLARIFY.value: [WorkflowNodeType.FINALIZER.value],
        },
    },
    RouteFunctionId.PLANNER.value: {
        "allowed_source_types": [WorkflowNodeType.PLANNER.value],
        "route_labels": [route.value for route in PlannerRoute],
        "target_types_by_label": {
            PlannerRoute.EXECUTE.value: [
                WorkflowNodeType.PARALLEL_DISPATCH.value,
                WorkflowNodeType.SERIAL_DISPATCH.value,
                WorkflowNodeType.RETRIEVAL_WORKER.value,
                WorkflowNodeType.THREAD_CONVERSATION_HISTORY_WORKER.value,
                WorkflowNodeType.DURABLE_MEMORY_WORKER.value,
                WorkflowNodeType.THREAD_EVENTS_WORKER.value,
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
    RouteFunctionId.PARALLEL_DISPATCH.value: {
        "allowed_source_types": [WorkflowNodeType.PARALLEL_DISPATCH.value],
        "route_labels": None,
        "target_types_by_label": None,
    },
    RouteFunctionId.SERIAL_DISPATCH.value: {
        "allowed_source_types": [WorkflowNodeType.SERIAL_DISPATCH.value],
        "route_labels": None,
        "target_types_by_label": None,
    },
    RouteFunctionId.ANSWER_QUALITY.value: {
        "allowed_source_types": [WorkflowNodeType.ANSWER_EVALUATOR.value],
        "route_labels": [route.value for route in AnswerQualityRoute],
        "target_types_by_label": {
            AnswerQualityRoute.PASS.value: [WorkflowNodeType.FINALIZER.value],
            AnswerQualityRoute.REVISE.value: [WorkflowNodeType.ANSWER_REVISER.value],
            AnswerQualityRoute.FINALIZE_CAUTIOUS.value: [WorkflowNodeType.FINALIZER.value],
        },
    },
    RouteFunctionId.CORRECTIVE_RETRIEVAL.value: {
        "allowed_source_types": [WorkflowNodeType.RETRIEVAL_QUALITY_GRADER.value],
        "route_labels": [route.value for route in CorrectiveRetrievalRoute],
        "target_types_by_label": {
            CorrectiveRetrievalRoute.SYNTHESIZE.value: [WorkflowNodeType.SYNTHESIZER.value],
            CorrectiveRetrievalRoute.CORRECT.value: [WorkflowNodeType.REPLANNER.value],
            CorrectiveRetrievalRoute.INSUFFICIENT.value: [WorkflowNodeType.SYNTHESIZER.value, WorkflowNodeType.FINALIZER.value],
        },
    },
    RouteFunctionId.GROUNDED_ANSWER.value: {
        "allowed_source_types": [WorkflowNodeType.GROUNDED_ANSWER_VERIFIER.value],
        "route_labels": [route.value for route in GroundedAnswerRoute],
        "target_types_by_label": {
            GroundedAnswerRoute.PASS.value: [WorkflowNodeType.FINALIZER.value],
            GroundedAnswerRoute.REVISE.value: [WorkflowNodeType.ANSWER_REVISER.value],
            GroundedAnswerRoute.CORRECT.value: [WorkflowNodeType.REPLANNER.value],
            GroundedAnswerRoute.FINALIZE_CAUTIOUS.value: [WorkflowNodeType.FINALIZER.value],
        },
    },
    RouteFunctionId.DEEP_TASK_DISPATCH.value: {
        "allowed_source_types": [WorkflowNodeType.DEEP_TASK_SCHEDULER.value],
        "route_labels": None,
        "target_types_by_label": None,
    },
    RouteFunctionId.DEEP_TASK.value: {
        "allowed_source_types": [WorkflowNodeType.DEEP_COORDINATOR.value],
        "route_labels": ["dispatch_more", "replan", "synthesize", "pause", "fail"],
        "target_types_by_label": {
            "dispatch_more": [WorkflowNodeType.DEEP_TASK_SCHEDULER.value],
            "replan": [WorkflowNodeType.DEEP_TASK_PLANNER.value],
            "synthesize": [WorkflowNodeType.DEEP_TASK_SYNTHESIZER.value],
            "pause": [WorkflowNodeType.DEEP_TASK_SCHEDULER.value, WorkflowNodeType.FINALIZER.value],
            "fail": [WorkflowNodeType.FINALIZER.value],
        },
    },
    RouteFunctionId.BUDGET_REVIEW.value: {
        "allowed_source_types": [WorkflowNodeType.EVIDENCE_CRITIC.value],
        "route_labels": ["continue", "steer", "accept_partial"],
        "target_types_by_label": {
            "continue": [WorkflowNodeType.DEEP_TASK_SCHEDULER.value],
            "steer": [WorkflowNodeType.DEEP_TASK_PLANNER.value],
            "accept_partial": [WorkflowNodeType.FINALIZER.value],
        },
    },
}

ROUTE_UI_OPTIONS: Dict[str, Dict[str, Dict[str, Any]]] = {
    RouteFunctionId.ROUTER.value: {
        RouterRoute.DOCUMENT.value: {"display_name": "Document question", "description": "Search uploaded documents.", "order": 0},
        RouterRoute.THREAD_CONVERSATION_HISTORY.value: {"display_name": "Thread Conversation History", "description": "Search prior messages in this thread.", "order": 1},
        RouterRoute.DURABLE_MEMORY.value: {"display_name": "Durable Memory", "description": "Recall durable user, project, or thread memory.", "order": 2},
        RouterRoute.THREAD_EVENTS.value: {"display_name": "Thread Events", "description": "Search chronological thread events.", "order": 3},
        RouterRoute.WEB.value: {"display_name": "Current information", "description": "Search approved external sources.", "order": 4},
        RouterRoute.COMPOUND.value: {"display_name": "Multi-source request", "description": "Escalate to a bounded retrieval plan.", "order": 5},
        RouterRoute.DIRECT.value: {"display_name": "Answer directly", "description": "Answer without retrieval.", "order": 6},
        RouterRoute.CLARIFY.value: {"display_name": "Needs clarification", "description": "Ask the user for more detail.", "order": 7},
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
    RouteFunctionId.ANSWER_QUALITY.value: {
        AnswerQualityRoute.PASS.value: {"display_name": "Quality passed", "description": "Finalize the reviewed answer.", "order": 0},
        AnswerQualityRoute.REVISE.value: {"display_name": "Revise once", "description": "Apply the bounded quality critique.", "order": 1},
        AnswerQualityRoute.FINALIZE_CAUTIOUS.value: {"display_name": "Finalize cautiously", "description": "Return the best available answer with limitations.", "order": 2},
    },
    RouteFunctionId.CORRECTIVE_RETRIEVAL.value: {
        CorrectiveRetrievalRoute.SYNTHESIZE.value: {"display_name": "Evidence is correct", "description": "Synthesize from eligible evidence.", "order": 0},
        CorrectiveRetrievalRoute.CORRECT.value: {"display_name": "Correct retrieval", "description": "Run another bounded retrieval wave.", "order": 1},
        CorrectiveRetrievalRoute.INSUFFICIENT.value: {"display_name": "Evidence insufficient", "description": "Finalize from verified evidence only.", "order": 2},
    },
    RouteFunctionId.GROUNDED_ANSWER.value: {
        GroundedAnswerRoute.PASS.value: {"display_name": "Grounding passed", "description": "Finalize the supported answer.", "order": 0},
        GroundedAnswerRoute.REVISE.value: {"display_name": "Revise answer", "description": "Apply one bounded answer revision.", "order": 1},
        GroundedAnswerRoute.CORRECT.value: {"display_name": "Retrieve missing support", "description": "Run another corrective retrieval wave.", "order": 2},
        GroundedAnswerRoute.FINALIZE_CAUTIOUS.value: {"display_name": "Finalize cautiously", "description": "Keep only verified claims and explicit gaps.", "order": 3},
    },
    RouteFunctionId.DEEP_TASK.value: {
        "dispatch_more": {"display_name": "Dispatch more", "description": "Run the next ready todo batch.", "order": 0},
        "replan": {"display_name": "Replan", "description": "Revise the bounded task plan.", "order": 1},
        "synthesize": {"display_name": "Synthesize", "description": "Build the final report.", "order": 2},
        "pause": {"display_name": "Pause", "description": "Checkpoint for human continuation.", "order": 3},
        "fail": {"display_name": "Fail", "description": "Finalize a terminal failure.", "order": 4},
    },
    RouteFunctionId.BUDGET_REVIEW.value: {
        "continue": {"display_name": "Continue research", "description": "Grant another budget tranche.", "order": 0},
        "steer": {"display_name": "Steer and continue", "description": "Apply guidance and grant another tranche.", "order": 1},
        "accept_partial": {"display_name": "Accept partial", "description": "Complete with the provisional answer.", "order": 2},
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
