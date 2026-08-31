from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.runnables import RunnableConfig

from langgraph_runtime.workflows.enums import RouterRoute
from langgraph_runtime.workflows.node_catalog import (
    node_type_default_max_visits,
    node_type_max_visits,
)


NODE_RUNTIME_CONFIG_KEY = "agent_workflow_node_runtime"


class WorkflowBudgetExceeded(RuntimeError):
    """Raised when a workflow reaches its configured visit budget."""

    def __init__(self, *, limit: int, node_id: str, node_type: str, visit_index: int,
                 observed_visits: int, run_id: str | None = None, thread_id: str | None = None) -> None:
        self.limit = limit
        self.node_id = node_id
        self.node_type = node_type
        self.visit_index = visit_index
        self.observed_visits = observed_visits
        self.run_id = run_id
        self.thread_id = thread_id
        super().__init__(f"Graph exceeded total visit limit {limit}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "exhausted", "limit": self.limit, "observed": self.observed_visits,
            "node_id": self.node_id, "node_type": self.node_type,
            "visit_index": self.visit_index, "run_id": self.run_id, "thread_id": self.thread_id,
        }


def _parallel_identity(value: Any) -> str:
    if isinstance(value, dict):
        if value.get("timeline_event_at"):
            source_record_id = value.get("message_id") or value.get("file_hash") or value.get("url") or value.get("title") or ""
            return f"timeline|{value.get('source_type') or ''}|{source_record_id}|{value.get('timeline_event_at')}"
        work_id = str(value.get("work_id") or "")
        attempt = str(value.get("attempt") or 1)
        if value.get("tool_name") or value.get("tool"):
            return "|".join(("tool", work_id, attempt, str(value.get("tool_name") or value.get("tool") or ""), str(value.get("invocation_sequence") or value.get("sequence") or 1)))
        if value.get("code") or value.get("error_code"):
            return "|".join(("error", work_id, attempt, str(value.get("code") or value.get("error_code") or "")))
        if work_id and any(key in value for key in ("name", "event", "event_name", "node", "node_id")):
            return "|".join((
                "event",
                work_id,
                attempt,
                str(value.get("node_id") or value.get("node") or value.get("worker_node_id") or ""),
                str(value.get("name") or value.get("event") or value.get("event_name") or value.get("status") or ""),
                str(value.get("visit_index") or value.get("node_visit_index") or 1),
            ))
        if value.get("node_id") or value.get("node"):
            return "|".join((
                "visit",
                str(value.get("node_id") or value.get("node") or ""),
                str(value.get("dispatch_id") or "serial"),
                work_id,
                attempt,
                str(value.get("visit_index") or value.get("node_visit_index") or 1),
            ))
        if work_id and value.get("status"):
            return "|".join(("result", work_id, attempt, str(value.get("status") or "")))
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def merge_parallel_deltas(left: List[Any], right: List[Any]) -> List[Any]:
    """Append immutable parallel deltas once across retries and checkpoint replay."""

    result = list(left or [])
    seen = {_parallel_identity(item) for item in result}
    for item in right or []:
        identity = _parallel_identity(item)
        if identity not in seen:
            result.append(item)
            seen.add(identity)
    return result


def merge_corrective_wave_records(left: List[Any], right: List[Any]) -> List[Any]:
    result = [dict(item) for item in left or [] if isinstance(item, dict)]
    positions = {str(item.get("record_id")): index for index, item in enumerate(result) if item.get("record_id")}
    for raw in right or []:
        if not isinstance(raw, dict) or not raw.get("record_id"):
            continue
        item = dict(raw)
        record_id = str(item["record_id"])
        if record_id in positions:
            result[positions[record_id]] = {**result[positions[record_id]], **item}
        else:
            positions[record_id] = len(result)
            result.append(item)
    return result


class WorkerWorkItem(TypedDict):
    dispatch_id: str
    dispatch_node_id: str
    dispatch_visit: int
    dispatch_started_epoch_ms: int
    dispatch_deadline_epoch_ms: int
    dispatch_proposal_signature: str
    work_id: str
    query_id: str
    ordinal: int
    worker_node_id: str
    worker_type: str
    query: str
    source_strategy: str
    source_scope: str
    source_expansion: bool
    evidence_kind: str
    dedupe_key: str
    attempt: int
    timeout_ms: int


class WorkerResultPacket(TypedDict):
    dispatch_id: str
    work_id: str
    ordinal: int
    worker_node_id: str
    worker_type: str
    attempt: int
    status: Literal["completed", "skipped", "failed", "timed_out", "cancelled"]
    evidence_packets: List[Dict[str, Any]]
    document_sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    chat_ids: List[str]
    memory_refs: List[Dict[str, Any]]
    node_events: List[Dict[str, Any]]
    tool_events: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    started_at: str
    completed_at: str
    elapsed_ms: float


class RouterRagState(TypedDict, total=False):
    agent_run_id: Optional[str]
    thread_id: str
    question: str
    llm_model: str
    embedding_model: str
    context_window: int
    use_web_search: bool
    use_reranker: bool
    bypass_clarification: bool
    system_role: str
    tool_instructions: Dict[str, str]
    custom_instructions: str
    client_timezone: Optional[str]
    client_locale: Optional[str]
    client_now_iso: Optional[str]
    transient_history_text: str
    pre_fetch_bundle: Dict[str, Any]
    prefetch_policy: Dict[str, Any]
    route: RouterRoute | str
    route_reason: str
    clarification_options: Optional[List[str]]
    evidence: str
    document_sources: List[Dict[str, Any]]
    web_sources: List[Dict[str, Any]]
    used_chat_ids: List[str]
    used_memory_ids: List[str]
    final_answer: str
    reasoning: str
    reasoning_available: bool
    reasoning_format: str
    human_review_decision: Dict[str, Any]
    hitl_policy: Dict[str, Any]
    hitl_decisions: List[Dict[str, Any]]
    hitl_approval_grants: Dict[str, Dict[str, Any]]
    hitl_gate_route: str
    hitl_gate_routes: Dict[str, Any]
    hitl_selected_options: Dict[str, List[str]]
    skipped_nodes: List[str]
    node_events: List[Dict[str, Any]]
    tool_events: List[Dict[str, Any]]
    errors: List[Dict[str, Any]]
    allowed_tool_ids: List[str]
    available_worker_nodes: List[Dict[str, Any]]
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
    parallel_policy: Dict[str, Any]
    parallel_enabled: bool
    parallel_runtime_override: bool
    parallel_aggregator_id: str
    dispatch_aggregator_id: str
    dispatch_mode: Literal["serial", "parallel"]
    dispatch_id: str
    dispatch_visit: int
    work_item: WorkerWorkItem
    work_items: List[WorkerWorkItem]
    work_item_proposals: List[Dict[str, Any]]
    worker_result_packets: Annotated[List[WorkerResultPacket], merge_parallel_deltas]
    parallel_evidence_deltas: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    parallel_document_source_deltas: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    parallel_web_source_deltas: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    parallel_chat_id_deltas: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    parallel_memory_ref_deltas: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    parallel_timeline_ref_deltas: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    parallel_node_event_deltas: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    parallel_tool_event_deltas: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    parallel_error_deltas: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    parallel_skipped_work_deltas: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    parallel_visit_records: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    parallel_attempt_records: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    parallel_summary: Dict[str, Any]
    dispatch_summary: Dict[str, Any]
    answer_quality_route: str
    answer_quality_report: Dict[str, Any]
    answer_revision_count: int
    corrective_policy: Dict[str, Any]
    corrective_wave: int
    corrective_history: List[Dict[str, Any]]
    corrective_wave_records: Annotated[List[Dict[str, Any]], merge_corrective_wave_records]
    corrective_policy_filtered_proposals: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    corrective_budget_usage: Dict[str, int]
    corrective_budget_exhausted_reason: str
    corrective_termination_reason: str
    retrieval_quality_report: Dict[str, Any]
    evidence_assessments: List[Dict[str, Any]]
    source_assessments: List[Dict[str, Any]]
    unresolved_gaps: List[str]
    corrective_retrieval_route: str
    grounding_report: Dict[str, Any]
    verified_claims: List[Dict[str, Any]]
    contradiction_report: List[Dict[str, Any]]
    grounded_answer_route: str
    agent_task_id: str
    task_version: int
    task_enabled_profiles: List[str]
    task_limits: Dict[str, Any]
    task_plan_revision: int
    task_run_plan_count: int
    task_plan: Dict[str, Any]
    task_todos: List[Dict[str, Any]]
    task_work_item: Dict[str, Any]
    task_work_items: List[Dict[str, Any]]
    task_result_packets: Annotated[List[Dict[str, Any]], merge_parallel_deltas]
    task_artifact_manifest: List[Dict[str, Any]]
    task_evidence_manifest: List[Dict[str, Any]]
    task_evidence_gaps: List[str]
    task_context_summary: Dict[str, Any]
    task_orchestration: Dict[str, Any]
    task_result_warnings: List[Dict[str, Any]]
    task_result_gaps: List[str]
    task_memory_snapshot: Dict[str, Any]
    task_budget_usage: Dict[str, Any]
    task_controller_route: str
    task_controller_reason: str
    task_pause_requested: bool
    task_cancel_requested: bool
    task_draft_metadata: Dict[str, Any]
    task_incomplete_reasons: List[str]
    task_critic_report: Dict[str, Any]
    web_search_mode: str
    task_web_access: str
    task_web_access_decision: Dict[str, Any]
    runtime_execution_mode: bool
    runtime_artifact_manifest: List[Dict[str, Any]]
    runtime_artifact_contents: Dict[str, str]
    runtime_artifacts: Annotated[List[Dict[str, Any]], merge_parallel_deltas]


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


def runtime_route_labels(config: Optional[RunnableConfig]) -> List[str]:
    runtime = node_runtime(config)
    labels = runtime.get("route_labels")
    return [str(item) for item in labels if isinstance(item, str) and item] if isinstance(labels, list) else []


def with_node_runtime_config(
    config: Optional[RunnableConfig],
    *,
    node_id: str,
    node_type: str,
    capabilities: List[str],
    visit_index: int,
    route_labels: Optional[List[str]] = None,
) -> RunnableConfig:
    updated = dict(config or {})
    configurable = dict(updated.get("configurable") or {})
    configurable[NODE_RUNTIME_CONFIG_KEY] = {
        "node_id": node_id,
        "node_type": node_type,
        "capabilities": list(capabilities),
        "visit_index": visit_index,
        "route_labels": list(route_labels or []),
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
        default_limit = int(
            node_type_default_max_visits(node_type)
            if node_type == "hitl_gate"
            else policy.get("default_max_node_visits", node_type_default_max_visits(node_type))
        )
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
        raise WorkflowBudgetExceeded(
            limit=total_limit,
            node_id=node_id,
            node_type=node_type,
            visit_index=visit_index,
            observed_visits=len(node_visit_sequence(state)),
            run_id=state.get("agent_run_id"),
            thread_id=state.get("thread_id"),
        )


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
