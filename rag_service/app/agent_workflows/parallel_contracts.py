from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Mapping

from app.agent_workflows.enums import EvidenceKind, WorkflowNodeType


PARALLEL_REFERENCE_WORKFLOW_ID = "orchestrator_worker_rag_agent"
PARALLEL_AUTHORIZED_WORKFLOW_IDS = frozenset({
    PARALLEL_REFERENCE_WORKFLOW_ID,
    "corrective_self_rag_agent",
})
PARALLEL_FEATURE_ENV = "ASKPDF_AGENT_WORKFLOW_PARALLEL_V1"
PARALLEL_EVENT_JOURNAL_LIMIT = 256
# The configured timeout is enforced inside the worker so it can become a
# reducer delta. LangGraph retains this slightly later outer watchdog.
PARALLEL_TIMEOUT_WATCHDOG_GRACE_MS = 250
PARALLEL_RETRY_INITIAL_INTERVAL_SECONDS = 0.25
PARALLEL_RETRY_BACKOFF_FACTOR = 2.0
PARALLEL_RETRY_MAX_INTERVAL_SECONDS = 1.0
PARALLEL_RETRY_JITTER = True

PARALLEL_POLICY_FIELDS: Dict[str, Dict[str, Any]] = {
    "enabled": {"type": "boolean", "default": True, "label": "Parallel execution"},
    "max_concurrency": {"type": "integer", "default": 4, "minimum": 1, "maximum": 16, "step": 1, "label": "Maximum concurrency"},
    "max_work_items": {"type": "integer", "default": 8, "minimum": 1, "maximum": 32, "step": 1, "label": "Maximum work items"},
    "dispatch_timeout_ms": {"type": "integer", "default": 60_000, "minimum": 1_000, "maximum": 300_000, "step": 1_000, "unit": "ms", "label": "Dispatch timeout"},
    "default_worker_timeout_ms": {"type": "integer", "default": 30_000, "minimum": 1_000, "maximum": 120_000, "step": 1_000, "unit": "ms", "label": "Worker timeout"},
    "web_worker_timeout_ms": {"type": "integer", "default": 45_000, "minimum": 1_000, "maximum": 180_000, "step": 1_000, "unit": "ms", "label": "Web worker timeout"},
    "max_attempts": {"type": "integer", "default": 2, "minimum": 1, "maximum": 5, "step": 1, "label": "Maximum attempts"},
    "minimum_successes": {"type": "integer", "default": 1, "minimum": 1, "maximum": 32, "step": 1, "label": "Minimum successes"},
    "continue_on_partial_failure": {"type": "boolean", "default": True, "label": "Continue with partial evidence"},
    "continue_on_insufficient_successes": {"type": "boolean", "default": False, "label": "Continue when no worker succeeds"},
}

DEFAULT_PARALLEL_POLICY: Dict[str, Any] = {
    key: descriptor["default"] for key, descriptor in PARALLEL_POLICY_FIELDS.items()
}
PARALLEL_POLICY_LIMITS = {
    key: (int(descriptor["minimum"]), int(descriptor["maximum"]))
    for key, descriptor in PARALLEL_POLICY_FIELDS.items()
    if descriptor["type"] == "integer"
}
PARALLEL_POLICY_BOOLEAN_FIELDS = frozenset(
    key for key, descriptor in PARALLEL_POLICY_FIELDS.items() if descriptor["type"] == "boolean"
)

PARALLEL_WORKER_EVIDENCE_KINDS = {
    WorkflowNodeType.RETRIEVAL_WORKER.value: EvidenceKind.DOCUMENT.value,
    WorkflowNodeType.THREAD_CONVERSATION_HISTORY_WORKER.value: EvidenceKind.THREAD_CONVERSATION_HISTORY.value,
    WorkflowNodeType.DURABLE_MEMORY_WORKER.value: EvidenceKind.DURABLE_MEMORY.value,
    WorkflowNodeType.THREAD_EVENTS_WORKER.value: EvidenceKind.THREAD_EVENTS.value,
    WorkflowNodeType.WEB_WORKER.value: EvidenceKind.WEB.value,
}
PARALLEL_RETRIEVAL_WORKER_TYPES = frozenset(PARALLEL_WORKER_EVIDENCE_KINDS)
PARALLEL_REDUCER_CHANNELS = (
    "worker_result_packets",
    "parallel_evidence_deltas",
    "parallel_document_source_deltas",
    "parallel_web_source_deltas",
    "parallel_chat_id_deltas",
    "parallel_memory_ref_deltas",
    "parallel_timeline_ref_deltas",
    "parallel_node_event_deltas",
    "parallel_tool_event_deltas",
    "parallel_error_deltas",
    "parallel_skipped_work_deltas",
    "parallel_visit_records",
    "parallel_attempt_records",
)
PARALLEL_FORBIDDEN_CUMULATIVE_CHANNELS = frozenset({
    "evidence", "evidence_packets", "document_sources", "web_sources",
    "used_chat_ids", "used_memory_ids", "node_events", "tool_events",
    "errors", "skipped_nodes", "node_visit_counts", "node_visit_sequence",
})

PARALLEL_TERMINAL_WORKER_STATUSES = frozenset({"completed", "skipped", "failed", "timed_out", "cancelled"})
PARALLEL_RETRYABLE_CLASSIFICATIONS = frozenset({"rate_limit", "network", "service_unavailable", "node_timeout"})
PARALLEL_NON_RETRYABLE_CLASSIFICATIONS = frozenset({"validation", "permission", "disabled_tool", "malformed_input", "cancelled", "contract", "dispatch_deadline"})

class ParallelEventName:
    DISPATCH_PLANNED = "dispatch.planned"
    DISPATCH_STARTED = "dispatch.started"
    WORKER_QUEUED = "worker.queued"
    WORKER_STARTED = "worker.started"
    WORKER_PROGRESS = "worker.progress"
    WORKER_RETRYING = "worker.retrying"
    WORKER_COMPLETED = "worker.completed"
    WORKER_SKIPPED = "worker.skipped"
    WORKER_FAILED = "worker.failed"
    WORKER_TIMED_OUT = "worker.timed_out"
    WORKER_CANCELLED = "worker.cancelled"
    BARRIER_REACHED = "dispatch.barrier_reached"
    AGGREGATION_COMPLETED = "aggregation.completed"
    AGGREGATION_PARTIAL = "aggregation.partial"
    DISPATCH_CANCELLED = "dispatch.cancelled"


PARALLEL_EVENT_NAMES = frozenset(
    value for key, value in vars(ParallelEventName).items() if key.isupper() and isinstance(value, str)
)
PARALLEL_EVENT_PREFIXES = ("dispatch.", "worker.", "aggregation.")
PARALLEL_WORKER_STATUS_BY_EVENT = {
    ParallelEventName.WORKER_QUEUED: "queued",
    ParallelEventName.WORKER_STARTED: "active",
    ParallelEventName.WORKER_PROGRESS: "active",
    ParallelEventName.WORKER_RETRYING: "retrying",
    ParallelEventName.WORKER_COMPLETED: "completed",
    ParallelEventName.WORKER_SKIPPED: "skipped",
    ParallelEventName.WORKER_FAILED: "failed",
    ParallelEventName.WORKER_TIMED_OUT: "timed_out",
    ParallelEventName.WORKER_CANCELLED: "cancelled",
}
PARALLEL_PROJECTED_WORKER_STATUSES = (
    "queued", "active", "retrying", "completed", "skipped", "failed", "timed_out", "cancelled",
)
PARALLEL_TERMINAL_EVENT_NAMES = frozenset(
    event_name
    for event_name, status in PARALLEL_WORKER_STATUS_BY_EVENT.items()
    if status in PARALLEL_TERMINAL_WORKER_STATUSES
)
PARALLEL_SUMMARY_COUNT_FIELDS = ("planned", "completed", "skipped", "failed", "timed_out", "cancelled", "retried")
PARALLEL_SUMMARY_METRIC_FIELDS = (
    "fan_out_width", "peak_concurrency", "evidence_packets_before_dedupe",
    "evidence_packets_after_dedupe", "document_sources_before_dedupe",
    "document_sources_after_dedupe", "web_sources_before_dedupe",
    "web_sources_after_dedupe",
)


def parallel_policy_catalog() -> Dict[str, Any]:
    return {
        "defaults": deepcopy(DEFAULT_PARALLEL_POLICY),
        "fields": deepcopy(PARALLEL_POLICY_FIELDS),
    }


def normalized_parallel_policy(value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, Mapping) else {}
    policy = deepcopy(DEFAULT_PARALLEL_POLICY)
    for key in PARALLEL_POLICY_BOOLEAN_FIELDS:
        if isinstance(raw.get(key), bool):
            policy[key] = raw[key]
    for key, (minimum, maximum) in PARALLEL_POLICY_LIMITS.items():
        try:
            parsed = int(raw.get(key, policy[key]))
        except (TypeError, ValueError):
            parsed = int(policy[key])
        policy[key] = max(minimum, min(parsed, maximum))
    policy["minimum_successes"] = min(policy["minimum_successes"], policy["max_work_items"])
    return policy


def parallel_timeout_watchdog_seconds(worker_timeout_ms: int, max_attempts: int) -> float:
    """Return a final watchdog long enough for all runtime-owned attempts."""
    attempts = max(1, int(max_attempts))
    retry_budget_seconds = PARALLEL_RETRY_MAX_INTERVAL_SECONDS * max(0, attempts - 1)
    return max(
        0.001,
        (max(1, int(worker_timeout_ms)) * attempts + PARALLEL_TIMEOUT_WATCHDOG_GRACE_MS) / 1000
        + retry_budget_seconds,
    )
