from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableConfig

from app.agent_workflows.enums import RouterRoute
from app.agent_workflows.node_catalog import (
    node_type_default_max_visits,
    node_type_max_visits,
)


NODE_RUNTIME_CONFIG_KEY = "agent_workflow_node_runtime"


class RouterRagState(TypedDict, total=False):
    agent_run_id: Optional[str]
    thread_id: str
    question: str
    llm_model: str
    embedding_model: str
    context_window: int
    use_web_search: bool
    use_reranker: bool
    system_role: str
    tool_instructions: Dict[str, str]
    custom_instructions: str
    client_timezone: Optional[str]
    client_locale: Optional[str]
    client_now_iso: Optional[str]
    pre_fetch_bundle: Dict[str, Any]
    route: RouterRoute | str
    route_reason: str
    clarification_options: Optional[List[str]]
    evidence: str
    document_sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    used_chat_ids: List[str]
    final_answer: str
    reasoning: str
    reasoning_available: bool
    reasoning_format: str
    human_review_decision: Dict[str, Any]
    hitl_policy: Dict[str, Any]
    hitl_decisions: List[Dict[str, Any]]
    hitl_gate_route: str
    hitl_gate_routes: Dict[str, Any]
    hitl_selected_options: Dict[str, List[str]]
    skipped_nodes: List[str]
    node_events: List[Dict[str, Any]]
    tool_events: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    allowed_tool_ids: List[str]
    workflow_id: str
    loop_policy: Dict[str, Any]
    node_visit_counts: Dict[str, int]
    node_visit_sequence: List[Dict[str, Any]]
    evidence_packets: List[Dict[str, Any]]
    hitl_interrupt_counts: Dict[str, int]
    execution_plan: List[str]
    replans: int
    replan_count: int
    replan_reason: str
    replan_history: List[Dict[str, Any]]
    evaluator_report: Dict[str, Any]
    evidence_gaps: List[str]
    evaluation_confidence: float
    evaluator_route: str


def node_runtime(config: Optional[RunnableConfig]) -> Dict[str, Any]:
    configurable = ((config or {}).get("configurable") or {})
    runtime = configurable.get(NODE_RUNTIME_CONFIG_KEY)
    return runtime if isinstance(runtime, dict) else {}


def runtime_node_id(config: Optional[RunnableConfig], fallback: str) -> str:
    runtime = node_runtime(config)
    node_id = runtime.get("node_id")
    return str(node_id) if isinstance(node_id, str) and node_id else fallback


def runtime_node_type(config: Optional[RunnableConfig], fallback: str) -> str:
    runtime = node_runtime(config)
    node_type = runtime.get("node_type")
    return str(node_type) if isinstance(node_type, str) and node_type else fallback


def runtime_node_capabilities(config: Optional[RunnableConfig]) -> List[str]:
    runtime = node_runtime(config)
    capabilities = runtime.get("capabilities")
    return [str(item) for item in capabilities] if isinstance(capabilities, list) else []


def runtime_visit_index(config: Optional[RunnableConfig]) -> Optional[int]:
    runtime = node_runtime(config)
    try:
        value = int(runtime.get("visit_index"))
    except (TypeError, ValueError):
        return None
    return value if value >= 1 else None


def with_node_runtime_config(
    config: Optional[RunnableConfig],
    *,
    node_id: str,
    node_type: str,
    capabilities: List[str],
    visit_index: int,
) -> RunnableConfig:
    updated = dict(config or {})
    configurable = dict(updated.get("configurable") or {})
    configurable[NODE_RUNTIME_CONFIG_KEY] = {
        "node_id": node_id,
        "node_type": node_type,
        "capabilities": list(capabilities),
        "visit_index": visit_index,
    }
    updated["configurable"] = configurable
    metadata = dict(updated.get("metadata") or {})
    metadata.update(
        {
            "node_id": node_id,
            "node_type": node_type,
            "node_capabilities": list(capabilities),
            "node_visit_index": visit_index,
        }
    )
    updated["metadata"] = metadata
    return updated


def loop_policy(state: RouterRagState) -> Dict[str, Any]:
    policy = state.get("loop_policy")
    return policy if isinstance(policy, dict) else {}


def node_visit_counts(state: RouterRagState) -> Dict[str, int]:
    counts = state.get("node_visit_counts")
    if not isinstance(counts, dict):
        return {}
    normalized: Dict[str, int] = {}
    for key, value in counts.items():
        if not isinstance(key, str):
            continue
        try:
            normalized[key] = max(0, int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def node_visit_sequence(state: RouterRagState) -> List[Dict[str, Any]]:
    sequence = state.get("node_visit_sequence")
    return [item for item in sequence if isinstance(item, dict)] if isinstance(sequence, list) else []


def node_visit_limit(state: RouterRagState, *, node_id: str, node_type: str) -> Optional[int]:
    policy = loop_policy(state)
    if not policy:
        return None
    node_limits = policy.get("node_visit_limits") if isinstance(policy.get("node_visit_limits"), dict) else {}
    if node_id in node_limits:
        try:
            return max(1, int(node_limits[node_id]))
        except (TypeError, ValueError):
            return 1
    try:
        default_limit = int(policy.get("default_max_node_visits", node_type_default_max_visits(node_type)))
    except (TypeError, ValueError):
        default_limit = node_type_default_max_visits(node_type)
    return max(1, min(default_limit, node_type_max_visits(node_type)))


def total_visit_limit(state: RouterRagState) -> Optional[int]:
    policy = loop_policy(state)
    if not policy:
        return None
    try:
        value = int(policy.get("max_total_visits"))
    except (TypeError, ValueError):
        return None
    return max(1, value)


def check_visit_budget(state: RouterRagState, *, node_id: str, node_type: str, visit_index: int) -> None:
    limit = node_visit_limit(state, node_id=node_id, node_type=node_type)
    if limit is not None and visit_index > limit:
        raise ValueError(f"Node {node_id} exceeded visit limit {limit}")
    total_limit = total_visit_limit(state)
    if total_limit is not None and len(node_visit_sequence(state)) + 1 > total_limit:
        raise ValueError(f"Graph exceeded total visit limit {total_limit}")


def with_visit_accounting(
    update: Dict[str, Any],
    state: RouterRagState,
    *,
    node_id: str,
    node_type: str,
    visit_index: int,
) -> Dict[str, Any]:
    counts = node_visit_counts(state)
    counts[node_id] = max(counts.get(node_id, 0), visit_index)
    sequence = [
        *node_visit_sequence(state),
        {"node": node_id, "node_type": node_type, "visit_index": visit_index},
    ]
    return {
        **update,
        "node_visit_counts": counts,
        "node_visit_sequence": sequence,
    }
