"""Central runtime transport mode configuration."""

from __future__ import annotations

import os
from enum import Enum


class AgentRuntimeMode(str, Enum):
    EXTERNAL = "external"
    IN_PROCESS = "in_process"


def agent_runtime_mode() -> AgentRuntimeMode:
    """Resolve runtime mode with external execution as the safe default."""

    configured = os.getenv("AGENT_RUNTIME_MODE", "").strip().lower()
    try:
        return AgentRuntimeMode(configured)
    except ValueError as exc:
        raise RuntimeError(
            "AGENT_RUNTIME_MODE must be 'external' or 'in_process'"
        ) from exc


def external_runtime_enabled() -> bool:
    return agent_runtime_mode() is AgentRuntimeMode.EXTERNAL
