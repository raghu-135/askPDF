"""Framework-neutral product projection vocabulary for parallel runtime events."""

PARALLEL_EVENT_JOURNAL_LIMIT = 256
PARALLEL_EVENT_PREFIXES = ("dispatch.", "worker.", "aggregation.", "corrective.")
PARALLEL_TERMINAL_WORKER_STATUSES = frozenset({"completed", "skipped", "failed", "timed_out", "cancelled"})
PARALLEL_PROJECTED_WORKER_STATUSES = (
    "queued", "active", "retrying", "completed", "skipped", "failed", "timed_out", "cancelled",
)
PARALLEL_SUMMARY_COUNT_FIELDS = ("planned", "completed", "skipped", "failed", "timed_out", "cancelled", "retried")
PARALLEL_SUMMARY_METRIC_FIELDS = (
    "fan_out_width", "peak_concurrency", "evidence_packets_before_dedupe",
    "evidence_packets_after_dedupe", "document_sources_before_dedupe",
    "document_sources_after_dedupe", "web_sources_before_dedupe",
    "web_sources_after_dedupe",
)


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
    value for key, value in vars(ParallelEventName).items()
    if key.isupper() and isinstance(value, str)
)
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
