"""Layer-specific runtime invocation context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from runtime_protocol.contracts import RuntimeTaskContext
from runtime_protocol.adapter import AgentRuntimeAdapter as RuntimeAdapter, AgentRuntimeEventSink


@dataclass(frozen=True)
class RuntimeExecutionContext:
    """In-process execution inputs kept outside the wire contract."""

    request: Any = None
    embedding_model: Optional[str] = None
    resolved_spec: Mapping[str, Any] = field(default_factory=dict)
    agent_run_context: Mapping[str, Any] = field(default_factory=dict)
    trace_recorder: Any = None
    cancellation_checker: Any = None
    pause_checker: Any = None
    pause_token_reader: Any = None
    pause_consumer: Any = None
    course_correction_reader: Any = None
    course_correction_acknowledger: Any = None
    task_id: Optional[str] = None
    task_worker_id: Optional[str] = None
    task_context: Optional[RuntimeTaskContext] = None
    operation_id: Optional[str] = None
    attempt_id: Optional[str] = None
    boundary_event_id: Optional[str] = None


class AgentRuntimeAdapter(RuntimeAdapter[RuntimeExecutionContext]):
    """Runtime SPI specialized with this layer's context."""

    async def prepare_execution_context(self, context: RuntimeExecutionContext) -> RuntimeExecutionContext:
        return context
