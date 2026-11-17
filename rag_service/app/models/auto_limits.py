"""Thread-level Auto orchestrator limits (defaults from environment)."""

from __future__ import annotations

import os
from typing import Any, Mapping

from pydantic import BaseModel, Field


def _env_int(name: str) -> int:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        raise ValueError(f"{name} environment variable is not set")
    return int(str(raw).strip())


def _env_bool(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        raise ValueError(f"{name} environment variable is not set")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class AutoLimitsSettings(BaseModel):
    max_total_visits: int = Field(ge=4, le=48)
    max_interrupts_per_run: int = Field(ge=1, le=20)
    max_delegates_per_turn: int = Field(ge=1, le=3)
    require_plan_approval_on_complex: bool = False
    require_plan_approval_all: bool = False


def default_auto_limits() -> dict[str, Any]:
    return AutoLimitsSettings(
      max_total_visits=_env_int("AUTO_MAX_TOTAL_VISITS"),
      max_interrupts_per_run=_env_int("AUTO_MAX_INTERRUPTS_PER_RUN"),
      max_delegates_per_turn=_env_int("AUTO_MAX_DELEGATES_PER_TURN"),
      require_plan_approval_on_complex=_env_bool("AUTO_REQUIRE_PLAN_APPROVAL_ON_COMPLEX"),
      require_plan_approval_all=_env_bool("AUTO_REQUIRE_PLAN_APPROVAL_ALL"),
    ).model_dump()


def merge_auto_limits(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    base = default_auto_limits()
    if not isinstance(overrides, Mapping):
        return base
    combined = {**base, **{k: v for k, v in overrides.items() if k in base}}
    if "max_delegates_per_turn" in combined:
        try:
            combined["max_delegates_per_turn"] = max(1, min(3, int(combined["max_delegates_per_turn"])))
        except (TypeError, ValueError):
            combined["max_delegates_per_turn"] = base["max_delegates_per_turn"]
    if "max_total_visits" in combined:
        try:
            combined["max_total_visits"] = max(4, min(48, int(combined["max_total_visits"])))
        except (TypeError, ValueError):
            combined["max_total_visits"] = base["max_total_visits"]
    if "max_interrupts_per_run" in combined:
        try:
            combined["max_interrupts_per_run"] = max(1, min(20, int(combined["max_interrupts_per_run"])))
        except (TypeError, ValueError):
            combined["max_interrupts_per_run"] = base["max_interrupts_per_run"]
    return AutoLimitsSettings(**combined).model_dump()