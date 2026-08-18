from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict

from app.agent_workflows.enums import WorkflowNodeType, WorkflowRuntimeKind

DEFAULT_AGENT_WORKFLOW_KEY_ENV = "ASKPDF_DEFAULT_AGENT_WORKFLOW_KEY"
DEFAULT_AGENT_WORKFLOW_KEY = "_".join((WorkflowNodeType.ROUTER.value, "rag", "agent"))
SUPPORTED_RUNTIME_KINDS = {kind.value for kind in WorkflowRuntimeKind}
RUNTIME_TEXT_FIELDS = {
    "label",
    "failure_code",
    "failure_reason_prefix",
    "success_context",
    "failure_context",
}
DEFAULT_COMPILED_RAG_RUNTIME = {
    "kind": WorkflowRuntimeKind.COMPILED_RAG.value,
    "label": "Compiled RAG",
    "failure_code": "compiled_rag_execution_failed",
    "failure_reason_prefix": "Exception during compiled RAG execution",
    "success_context": "Context retrieved by compiled RAG workflow.",
    "failure_context": "Compiled RAG workflow execution failed gracefully.",
    "features": {"supports_replans": False},
    "prompt_preview": WorkflowNodeType.ROUTER.value,
}
ALLOWED_WORKFLOW_CONFIG_KEYS = {
    "use_web_search",
    "use_reranker",
    "system_role",
    "tool_instructions",
    "custom_instructions",
    "allowed_tool_ids",
    "prefetch_policy",
    "hitl_policy",
    "replans",
    "graph",
    "context_policy",
    "loop_policy",
    "builder_ui",
    "parallel_policy",
    "corrective_policy",
    "task_policy",
}


def default_agent_workflow_key() -> str:
    return os.getenv(DEFAULT_AGENT_WORKFLOW_KEY_ENV, DEFAULT_AGENT_WORKFLOW_KEY).strip() or DEFAULT_AGENT_WORKFLOW_KEY


def workflow_runtime(spec: Dict[str, Any]) -> Dict[str, Any]:
    runtime = spec.get("runtime") if isinstance(spec.get("runtime"), dict) else {}
    return runtime if isinstance(runtime, dict) else {}


def with_default_runtime(spec: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(spec)
    if not isinstance(normalized.get("runtime"), dict):
        normalized["runtime"] = deepcopy(DEFAULT_COMPILED_RAG_RUNTIME)
    return normalized


def workflow_runtime_features(spec: Dict[str, Any]) -> Dict[str, Any]:
    features = workflow_runtime(spec).get("features")
    return features if isinstance(features, dict) else {}


def workflow_supports_replans(spec: Dict[str, Any]) -> bool:
    return bool(workflow_runtime_features(spec).get("supports_replans"))


def workflow_allows_replans_override(spec: Dict[str, Any]) -> bool:
    """Return whether generic thread/request replan settings apply to this workflow."""

    features = workflow_runtime_features(spec)
    return (
        bool(features.get("supports_replans"))
        and not bool(features.get("supports_corrective_retrieval"))
        and features.get("allows_replans_override", True) is not False
    )


def repeatable_node_types_for_replans(spec: Dict[str, Any]) -> set[str]:
    config = spec.get("config") if isinstance(spec.get("config"), dict) else {}
    graph = config.get("graph") if isinstance(config.get("graph"), dict) else {}
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
    return {
        str(node.get("type"))
        for node in nodes
        if isinstance(node, dict)
        and isinstance(node.get("type"), str)
        and node.get("type")
        in {
            WorkflowNodeType.RETRIEVAL_WORKER.value,
            WorkflowNodeType.THREAD_CONVERSATION_HISTORY_WORKER.value,
            WorkflowNodeType.DURABLE_MEMORY_WORKER.value,
            WorkflowNodeType.THREAD_EVENTS_WORKER.value,
            WorkflowNodeType.WEB_WORKER.value,
            WorkflowNodeType.EVIDENCE_EVALUATOR.value,
            WorkflowNodeType.RETRIEVAL_QUALITY_GRADER.value,
        }
    }


def replan_loop_policy(spec: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    # Import lazily because node_catalog depends on model configuration, which
    # imports this module during application startup.
    from app.agent_workflows.node_catalog import node_type_max_visits

    graph = config.get("graph") if isinstance(config.get("graph"), dict) else {}
    node_types = {
        str(node.get("id")): str(node.get("type"))
        for node in graph.get("nodes", [])
        if isinstance(node, dict) and node.get("id") and node.get("type")
    }
    try:
        replans = max(1, int(config.get("replans", 1)))
    except (TypeError, ValueError):
        replans = 1
    repeatable_node_types = repeatable_node_types_for_replans({**spec, "config": config})
    node_visit_limits = {
        node_id: replans + 1
        for node_id, node_type in node_types.items()
        if node_type in repeatable_node_types
    }
    task_policy = config.get("task_policy") if isinstance(config.get("task_policy"), dict) else {}
    task_limits = task_policy.get("limits") if isinstance(task_policy.get("limits"), dict) else {}
    try:
        max_todos = max(1, min(50, int(task_limits.get("max_todos", 50))))
        max_attempts = max(1, min(2, int(task_limits.get("max_attempts_per_todo", 2))))
        max_interrupts = max(1, min(16, int(((config.get("hitl_policy") or {}).get("max_interrupts_per_run", 16)))))
    except (TypeError, ValueError):
        max_todos, max_attempts, max_interrupts = 50, 2, 16
    deep_control_visits = max_todos + replans + 1
    for node_id, node_type in node_types.items():
        if node_type == WorkflowNodeType.REPLANNER.value:
            node_visit_limits[node_id] = replans
        elif node_type == WorkflowNodeType.SERIAL_DISPATCH.value:
            worker_count = sum(
                1 for value in node_types.values()
                if value in {
                    WorkflowNodeType.RETRIEVAL_WORKER.value,
                    WorkflowNodeType.THREAD_CONVERSATION_HISTORY_WORKER.value,
                    WorkflowNodeType.DURABLE_MEMORY_WORKER.value,
                    WorkflowNodeType.THREAD_EVENTS_WORKER.value,
                    WorkflowNodeType.WEB_WORKER.value,
                }
            )
            generated_limit = (worker_count + 1) * (replans + 1)
            # Keep generated budgets within the authoritative node contract.
            # Otherwise resolver output can invalidate an otherwise valid
            # built-in workflow before compilation.
            node_visit_limits[node_id] = min(generated_limit, node_type_max_visits(node_type))
        elif node_type == WorkflowNodeType.AGGREGATOR.value:
            node_visit_limits[node_id] = replans + 1
        elif node_type == WorkflowNodeType.PARALLEL_DISPATCH.value:
            node_visit_limits[node_id] = replans + 1
        elif node_type == WorkflowNodeType.GROUNDED_ANSWER_VERIFIER.value:
            node_visit_limits[node_id] = replans + 2
        elif node_type == WorkflowNodeType.ANSWER_EVALUATOR.value:
            node_visit_limits[node_id] = 2
        elif node_type == WorkflowNodeType.ANSWER_REVISER.value:
            node_visit_limits[node_id] = 1
        elif node_type == WorkflowNodeType.DEEP_TASK_PLANNER.value:
            node_visit_limits[node_id] = replans + 1
        elif node_type in {
            WorkflowNodeType.DEEP_TASK_SCHEDULER.value,
            WorkflowNodeType.DEEP_COORDINATOR.value,
        }:
            node_visit_limits[node_id] = deep_control_visits
        elif node_type == WorkflowNodeType.DEEP_RESEARCH_SUBAGENT.value:
            node_visit_limits[node_id] = max_todos * max_attempts
        elif node_type == WorkflowNodeType.HITL_GATE.value and workflow_runtime_features(spec).get("supports_long_running_tasks"):
            node_visit_limits[node_id] = max_interrupts
    max_total_visits = sum(node_visit_limits.get(node_id, 1) for node_id in node_types)
    return {
        "max_total_visits": max_total_visits,
        "default_max_node_visits": 1,
        "node_visit_limits": {node_id: node_visit_limits[node_id] for node_id in sorted(node_visit_limits)},
    }


def runtime_execution_options(spec: Dict[str, Any]) -> Dict[str, str]:
    runtime = workflow_runtime(spec)
    options = {field: str(runtime.get(field) or "") for field in RUNTIME_TEXT_FIELDS}
    label = options["label"] or "Compiled RAG"
    defaults = {
        "label": label,
        "failure_code": "compiled_rag_execution_failed",
        "failure_reason_prefix": f"Exception during {label} execution",
        "success_context": f"Context retrieved by {label}.",
        "failure_context": f"{label} execution failed gracefully.",
    }
    return {field: options[field] or defaults[field] for field in RUNTIME_TEXT_FIELDS}


def normalize_runtime_for_validation(runtime: Any) -> Dict[str, Any]:
    return deepcopy(runtime) if isinstance(runtime, dict) else {}
