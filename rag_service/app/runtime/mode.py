"""Central runtime transport mode configuration."""

from __future__ import annotations

import logging
import os
from enum import Enum


logger = logging.getLogger(__name__)


class AgentRuntimeMode(str, Enum):
    EXTERNAL = "external"
    IN_PROCESS = "in_process"


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def agent_runtime_mode() -> AgentRuntimeMode:
    """Resolve runtime mode with external execution as the safe default."""

    configured = os.getenv("AGENT_RUNTIME_MODE")
    if configured is not None:
        value = configured.strip().lower()
        try:
            return AgentRuntimeMode(value)
        except ValueError as exc:
            raise RuntimeError(
                "AGENT_RUNTIME_MODE must be 'external' or 'in_process'"
            ) from exc

    legacy = os.getenv("AGENT_RUNTIME_EXTERNAL_ENABLED")
    if legacy is None:
        return AgentRuntimeMode.EXTERNAL

    value = legacy.strip().lower()
    logger.warning(
        "AGENT_RUNTIME_EXTERNAL_ENABLED is deprecated; use "
        "AGENT_RUNTIME_MODE=external|in_process"
    )
    if value in _TRUE_VALUES:
        return AgentRuntimeMode.EXTERNAL
    if value in _FALSE_VALUES:
        return AgentRuntimeMode.IN_PROCESS
    raise RuntimeError(
        "AGENT_RUNTIME_EXTERNAL_ENABLED must be a boolean value when used"
    )


def external_runtime_enabled() -> bool:
    return agent_runtime_mode() is AgentRuntimeMode.EXTERNAL
