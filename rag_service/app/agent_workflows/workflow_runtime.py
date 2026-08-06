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
        }
    }


def replan_loop_policy(spec: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
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
    for node_id, node_type in node_types.items():
        if node_type == WorkflowNodeType.REPLANNER.value:
            node_visit_limits[node_id] = replans
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
