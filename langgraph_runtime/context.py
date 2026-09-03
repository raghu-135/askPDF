"""Runtime-local execution context reconstructed from the JSON wire contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from runtime_protocol.contracts import RuntimeTaskContext


@dataclass(frozen=True)
class RuntimeExecutionContext:
    request: Any = None
    embedding_model: str | None = None
    resolved_spec: Mapping[str, Any] = field(default_factory=dict)
    agent_run_context: Mapping[str, Any] = field(default_factory=dict)
    trace_recorder: Any = None
    cancellation_checker: Any = None
    pause_checker: Any = None
    course_correction_reader: Any = None
    course_correction_acknowledger: Any = None
    task_id: str | None = None
    task_worker_id: str | None = None
    task_context: RuntimeTaskContext | None = None
    operation_id: str | None = None
    attempt_id: str | None = None
    boundary_event_id: str | None = None
