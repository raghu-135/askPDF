"""Context packing ratios shared by control plane and LangGraph runtime."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

CHARS_PER_TOKEN = 4
_FALLBACK_CONTEXT_WINDOW = 8192

_CONTEXT_ENV_DEFAULTS: dict[str, str] = {
    "CONTEXT_USABLE_RATIO": "0.80",
    "CONTEXT_RECENT_RATIO": "0.22",
    "CONTEXT_SEMANTIC_RATIO": "0.12",
    "CONTEXT_DURABLE_MEMORY_RATIO": "0.06",
    "CONTEXT_DOCUMENT_RATIO": "0.28",
    "CONTEXT_MEMORY_EXPANDED_RATIO": "0.12",
    "CONTEXT_EVIDENCE_RATIO": "0.45",
    "CONTEXT_DRAFT_RATIO": "0.25",
    "CONTEXT_PLAN_SECTION_RATIO": "0.20",
    "CONTEXT_PLAN_TOTAL_RATIO": "0.50",
    "CONTEXT_PLAN_MEMORY_RATIO": "0.12",
}

_PREFETCH_KEYS = (
    "recent_ratio",
    "semantic_ratio",
    "durable_memory_ratio",
    "document_ratio",
)

_OVERRIDE_ALIASES = {
    "usable": "usable_ratio",
    "usable_ratio": "usable_ratio",
    "recent": "recent_ratio",
    "recent_ratio": "recent_ratio",
    "semantic": "semantic_ratio",
    "semantic_ratio": "semantic_ratio",
    "durable_memory": "durable_memory_ratio",
    "durable_memory_ratio": "durable_memory_ratio",
    "document": "document_ratio",
    "document_ratio": "document_ratio",
    "memory_expanded": "memory_expanded_ratio",
    "memory_expanded_ratio": "memory_expanded_ratio",
}


@dataclass(frozen=True)
class ContextBudgetSettings:
    usable_ratio: float
    recent_ratio: float
    semantic_ratio: float
    durable_memory_ratio: float
    document_ratio: float
    memory_expanded_ratio: float
    evidence_ratio: float
    draft_ratio: float
    plan_section_ratio: float
    plan_total_ratio: float
    plan_memory_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "usable_ratio": self.usable_ratio,
            "recent_ratio": self.recent_ratio,
            "semantic_ratio": self.semantic_ratio,
            "durable_memory_ratio": self.durable_memory_ratio,
            "document_ratio": self.document_ratio,
            "memory_expanded_ratio": self.memory_expanded_ratio,
            "evidence_ratio": self.evidence_ratio,
            "draft_ratio": self.draft_ratio,
            "plan_section_ratio": self.plan_section_ratio,
            "plan_total_ratio": self.plan_total_ratio,
            "plan_memory_ratio": self.plan_memory_ratio,
        }

    def prefetch_sum(self) -> float:
        return self.recent_ratio + self.semantic_ratio + self.durable_memory_ratio + self.document_ratio


def _parse_open_ratio(value: str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite ratio in (0, 1]") from exc
    if not math.isfinite(parsed) or not 0 < parsed <= 1:
        raise ValueError(f"{name} must be a finite ratio in (0, 1]")
    return parsed


def _ratio_from_env(name: str, values: Mapping[str, str]) -> float:
    raw = values.get(name)
    if raw is None or not str(raw).strip():
        raise ValueError(f"{name} is required")
    return _parse_open_ratio(str(raw).strip(), name=name)


def default_context_budget_settings() -> ContextBudgetSettings:
    """Return installed defaults without requiring environment variables."""

    return ContextBudgetSettings(
        usable_ratio=float(_CONTEXT_ENV_DEFAULTS["CONTEXT_USABLE_RATIO"]),
        recent_ratio=float(_CONTEXT_ENV_DEFAULTS["CONTEXT_RECENT_RATIO"]),
        semantic_ratio=float(_CONTEXT_ENV_DEFAULTS["CONTEXT_SEMANTIC_RATIO"]),
        durable_memory_ratio=float(_CONTEXT_ENV_DEFAULTS["CONTEXT_DURABLE_MEMORY_RATIO"]),
        document_ratio=float(_CONTEXT_ENV_DEFAULTS["CONTEXT_DOCUMENT_RATIO"]),
        memory_expanded_ratio=float(_CONTEXT_ENV_DEFAULTS["CONTEXT_MEMORY_EXPANDED_RATIO"]),
        evidence_ratio=float(_CONTEXT_ENV_DEFAULTS["CONTEXT_EVIDENCE_RATIO"]),
        draft_ratio=float(_CONTEXT_ENV_DEFAULTS["CONTEXT_DRAFT_RATIO"]),
        plan_section_ratio=float(_CONTEXT_ENV_DEFAULTS["CONTEXT_PLAN_SECTION_RATIO"]),
        plan_total_ratio=float(_CONTEXT_ENV_DEFAULTS["CONTEXT_PLAN_TOTAL_RATIO"]),
        plan_memory_ratio=float(_CONTEXT_ENV_DEFAULTS["CONTEXT_PLAN_MEMORY_RATIO"]),
    )


def load_context_budget_settings(environ: Mapping[str, str] | None = None) -> ContextBudgetSettings:
    values = dict(os.environ if environ is None else environ)
    return ContextBudgetSettings(
        usable_ratio=_ratio_from_env("CONTEXT_USABLE_RATIO", values),
        recent_ratio=_ratio_from_env("CONTEXT_RECENT_RATIO", values),
        semantic_ratio=_ratio_from_env("CONTEXT_SEMANTIC_RATIO", values),
        durable_memory_ratio=_ratio_from_env("CONTEXT_DURABLE_MEMORY_RATIO", values),
        document_ratio=_ratio_from_env("CONTEXT_DOCUMENT_RATIO", values),
        memory_expanded_ratio=_ratio_from_env("CONTEXT_MEMORY_EXPANDED_RATIO", values),
        evidence_ratio=_ratio_from_env("CONTEXT_EVIDENCE_RATIO", values),
        draft_ratio=_ratio_from_env("CONTEXT_DRAFT_RATIO", values),
        plan_section_ratio=_ratio_from_env("CONTEXT_PLAN_SECTION_RATIO", values),
        plan_total_ratio=_ratio_from_env("CONTEXT_PLAN_TOTAL_RATIO", values),
        plan_memory_ratio=_ratio_from_env("CONTEXT_PLAN_MEMORY_RATIO", values),
    )


def validate_prefetch_ratios(settings: ContextBudgetSettings) -> None:
    total = settings.prefetch_sum()
    if total > 1.0:
        raise ValueError(
            "Prefetch context ratios must sum to at most 1 "
            f"(recent + semantic + durable_memory + document = {total:g})"
        )


def _normalize_override_map(overrides: Mapping[str, Any] | None) -> dict[str, float]:
    if not overrides:
        return {}
    normalized: dict[str, float] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        target = _OVERRIDE_ALIASES.get(str(key))
        if target is None:
            continue
        normalized[target] = _parse_open_ratio(value, name=str(key))
    return normalized


def resolve_context_budget(overrides: Mapping[str, Any] | None = None, *, environ: Mapping[str, str] | None = None) -> ContextBudgetSettings:
    base = load_context_budget_settings(environ)
    merged = base.to_dict()
    merged.update(_normalize_override_map(overrides))
    settings = ContextBudgetSettings(**merged)
    validate_prefetch_ratios(settings)
    return settings


def slice_ratio(settings: ContextBudgetSettings | Mapping[str, Any], key: str) -> float:
    if isinstance(settings, ContextBudgetSettings):
        return float(getattr(settings, key))
    return float(settings[key])


def usable_context_chars(
    context_window: int | None,
    *,
    ratio: float | None = None,
    ratio_key: str | None = None,
    budget: ContextBudgetSettings | Mapping[str, Any] | None = None,
) -> int:
    settings = budget if budget is not None else load_context_budget_settings()
    try:
        tokens = int(context_window) if context_window is not None else 0
    except (TypeError, ValueError):
        tokens = 0
    if tokens <= 0:
        tokens = _FALLBACK_CONTEXT_WINDOW
    usable_ratio = slice_ratio(settings, "usable_ratio")
    share = ratio
    if share is None and ratio_key:
        share = slice_ratio(settings, ratio_key)
    if share is None:
        share = 1.0
    usable = max(1, int(tokens * usable_ratio * CHARS_PER_TOKEN))
    return max(1, int(usable * share))


# Item-count helpers for prefetch (mirrors llm_server_client defaults).
AVG_CHUNK_CHARS = 500
AVG_TURN_CHARS = 600


def compute_prefetch_budget(
    context_window: int,
    budget: ContextBudgetSettings | Mapping[str, Any] | None = None,
) -> dict[str, int]:
    settings = budget if budget is not None else load_context_budget_settings()
    if isinstance(settings, Mapping):
        settings = ContextBudgetSettings(**dict(settings))
    usable = int(int(context_window) * settings.usable_ratio * CHARS_PER_TOKEN)
    durable_memory_chars = max(1, int(usable * settings.durable_memory_ratio))
    return {
        "recent_history_chars": int(usable * settings.recent_ratio),
        "semantic_history_chars": int(usable * settings.semantic_ratio),
        "durable_memory_chars": durable_memory_chars,
        "document_context_chars": int(usable * settings.document_ratio),
        "document_limit": max(3, int(usable * settings.document_ratio) // AVG_CHUNK_CHARS),
        "semantic_limit": max(3, int(usable * settings.semantic_ratio) // AVG_CHUNK_CHARS),
        "durable_memory_limit": max(1, math.ceil(durable_memory_chars / AVG_CHUNK_CHARS) * 2),
        "recent_turn_limit": max(4, int(usable * settings.recent_ratio) // AVG_TURN_CHARS),
    }


def context_budget_env_defaults() -> dict[str, str]:
    return dict(_CONTEXT_ENV_DEFAULTS)
