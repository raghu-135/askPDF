"""Layer-specific runtime invocation context."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from runtime_protocol.contracts import RuntimeTaskContext
from runtime_protocol.adapter import AgentRuntimeAdapter as RuntimeAdapter, AgentRuntimeEventSink


@dataclass(frozen=True)
class RuntimeInvocationContext:
    """Product-owned values that may be serialized onto the runtime wire."""

    request_payload: Mapping[str, Any] = field(default_factory=dict)
    embedding_model: Optional[str] = None
    resolved_spec: Mapping[str, Any] = field(default_factory=dict)
    agent_run_context: Mapping[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    task_worker_id: Optional[str] = None
    task_context: Optional[RuntimeTaskContext] = None


class AgentRuntimeAdapter(RuntimeAdapter[RuntimeInvocationContext]):
    """Runtime SPI specialized with this layer's context."""

    pass
