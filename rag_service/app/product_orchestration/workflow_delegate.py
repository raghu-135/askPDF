"""Auto workflow delegate catalog resolution and admission."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional

from app.product_orchestration.workflow_runtime import workflow_is_chat_eligible

AUTO_WORKFLOW_ID = "auto_rag_agent"
HERMES_WORKFLOW_ID = "hermes_rag_agent"
DEEP_RESEARCH_WORKFLOW_ID = "deep_research_agent"


def normalized_auto_policy(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    policy = config.get("auto_policy") if isinstance(config, Mapping) else {}
    if not isinstance(policy, dict):
        policy = {}
    delegate_policy = policy.get("delegate_policy") if isinstance(policy.get("delegate_policy"), dict) else {}
    exclude = {
        str(item).strip()
        for item in delegate_policy.get("exclude_workflow_ids") or [AUTO_WORKFLOW_ID, HERMES_WORKFLOW_ID]
        if str(item).strip()
    }
    exclude.add(AUTO_WORKFLOW_ID)
    return {
        "require_plan_approval_on_complex": bool(policy.get("require_plan_approval_on_complex", True)),
        "max_delegates_per_turn": max(1, int(policy.get("max_delegates_per_turn", 1) or 1)),
        "pass_prompt_overrides": policy.get("pass_prompt_overrides", True) is not False,
        "delegate_policy": {
            "mode": str(delegate_policy.get("mode") or "catalog"),
            "include_custom": delegate_policy.get("include_custom", True) is not False,
            "frameworks": [
                str(item) for item in (delegate_policy.get("frameworks") or ["langgraph"]) if str(item).strip()
            ],
            "exclude_workflow_ids": sorted(exclude),
        },
    }


def _workflow_admission_row(workflow: Any, *, auto_policy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if workflow is None:
        return None
    workflow_id = str(getattr(workflow, "id", "") or "").strip()
    if not workflow_id:
        return None
    spec = workflow.spec_json if isinstance(getattr(workflow, "spec_json", None), dict) else {}
    exclude = set(auto_policy["delegate_policy"]["exclude_workflow_ids"])
    if workflow_id in exclude:
        return None
    framework = str(getattr(workflow, "framework", "") or "").strip()
    allowed_frameworks = set(auto_policy["delegate_policy"]["frameworks"])
    if framework not in allowed_frameworks:
        return None
    features = (spec.get("runtime") or {}).get("features") or {}
    if workflow_id == DEEP_RESEARCH_WORKFLOW_ID:
        if not features.get("supports_long_running_tasks"):
            return None
    elif not workflow_is_chat_eligible(spec):
        return None
    validation = workflow.validation_result_json if isinstance(getattr(workflow, "validation_result_json", None), dict) else {}
    if validation and validation.get("valid") is False:
        return None
    visibility = str(getattr(workflow, "visibility", "") or "")
    if visibility == "deleted":
        return None
    runtime = spec.get("runtime") if isinstance(spec.get("runtime"), dict) else {}
    runtime_features = runtime.get("features") if isinstance(runtime.get("features"), dict) else {}
    return {
        "workflow_id": workflow_id,
        "name": str(getattr(workflow, "name", "") or workflow_id),
        "description": str(getattr(workflow, "description", "") or ""),
        "category": getattr(workflow, "category", None),
        "framework": framework,
        "supports_long_running_tasks": bool(runtime_features.get("supports_long_running_tasks")),
        "supports_replans": bool(runtime_features.get("supports_replans")),
    }


async def resolve_delegate_catalog(
    repository: Any,
    *,
    auto_policy: Mapping[str, Any] | None = None,
    include_custom: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    policy = normalized_auto_policy({"auto_policy": auto_policy} if auto_policy else {})
    if include_custom is None:
        include_custom = bool(policy["delegate_policy"]["include_custom"])
    await repository.seed_builtin_workflows()
    workflows = await repository.list_workflows(include_custom=include_custom)
    catalog: List[Dict[str, Any]] = []
    for workflow in workflows:
        row = _workflow_admission_row(workflow, auto_policy=policy)
        if row is not None:
            catalog.append(row)
    if not any(row["workflow_id"] == DEEP_RESEARCH_WORKFLOW_ID for row in catalog):
        deep = await repository.get_workflow(DEEP_RESEARCH_WORKFLOW_ID, include_custom=False)
        row = _workflow_admission_row(deep, auto_policy=policy)
        if row is not None:
            catalog.append(row)
    catalog.sort(key=lambda item: (item.get("name") or item["workflow_id"]).lower())
    return catalog


def inject_delegate_catalog_into_spec(
    spec: Dict[str, Any],
    catalog: List[Dict[str, Any]],
) -> Dict[str, Any]:
    materialized = deepcopy(spec)
    config = materialized.get("config") if isinstance(materialized.get("config"), dict) else {}
    materialized["config"] = config
    config["available_delegate_workflows"] = [deepcopy(item) for item in catalog]
    return materialized


def catalog_workflow_ids(catalog: List[Dict[str, Any]]) -> set[str]:
    return {str(item.get("workflow_id") or "") for item in catalog if item.get("workflow_id")}
