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
def default_agent_workflow_key() -> str:
    return os.getenv(DEFAULT_AGENT_WORKFLOW_KEY_ENV, DEFAULT_AGENT_WORKFLOW_KEY).strip() or DEFAULT_AGENT_WORKFLOW_KEY


def workflow_runtime(spec: Dict[str, Any]) -> Dict[str, Any]:
    runtime = spec.get("runtime") if isinstance(spec.get("runtime"), dict) else {}
    return runtime if isinstance(runtime, dict) else {}


def workflow_runtime_features(spec: Dict[str, Any]) -> Dict[str, Any]:
    features = workflow_runtime(spec).get("features")
    return features if isinstance(features, dict) else {}


def workflow_supports_replans(spec: Dict[str, Any]) -> bool:
    return bool(workflow_runtime_features(spec).get("supports_replans"))


def workflow_supports_long_running_tasks(spec: Dict[str, Any]) -> bool:
    return bool(workflow_runtime_features(spec).get("supports_long_running_tasks"))


def workflow_is_chat_eligible(spec: Dict[str, Any]) -> bool:
    """Return whether a workflow may be selected for ordinary thread chat."""

    return not workflow_supports_long_running_tasks(spec)


def workflow_allows_replans_override(spec: Dict[str, Any]) -> bool:
    """Return whether generic thread/request replan settings apply to this workflow."""

    features = workflow_runtime_features(spec)
    return (
        bool(features.get("supports_replans"))
        and not bool(features.get("supports_corrective_retrieval"))
        and features.get("allows_replans_override", True) is not False
    )


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
