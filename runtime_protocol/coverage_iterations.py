"""Coverage iteration limits shared by control plane and LangGraph runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

_COVERAGE_KEYS = (
    "max_iterations_sufficient_coverage",
    "max_iterations_probably_sufficient_coverage",
    "web_search_iteration_bonus",
)

_ENV_BY_KEY = {
    "max_iterations_sufficient_coverage": "MAX_ITERATIONS_SUFFICIENT_COVERAGE",
    "max_iterations_probably_sufficient_coverage": "MAX_ITERATIONS_PROBABLY_SUFFICIENT_COVERAGE",
    "web_search_iteration_bonus": "WEB_SEARCH_ITERATION_BONUS",
}

_BOUNDS = {
    "max_iterations_sufficient_coverage": (1, 64),
    "max_iterations_probably_sufficient_coverage": (1, 64),
    "web_search_iteration_bonus": (0, 32),
}


def _parse_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return parsed


def _env_int(name: str, environ: Mapping[str, str]) -> int:
    raw = environ.get(name)
    if raw is None or not str(raw).strip():
        raise ValueError(f"{name} is required")
    return _parse_int(str(raw).strip(), name=name)


@dataclass(frozen=True)
class CoverageIterationSettings:
    max_iterations_sufficient_coverage: int
    max_iterations_probably_sufficient_coverage: int
    web_search_iteration_bonus: int

    def to_dict(self) -> dict[str, int]:
        return {
            "max_iterations_sufficient_coverage": self.max_iterations_sufficient_coverage,
            "max_iterations_probably_sufficient_coverage": self.max_iterations_probably_sufficient_coverage,
            "web_search_iteration_bonus": self.web_search_iteration_bonus,
        }


def load_coverage_iteration_settings(environ: Mapping[str, str] | None = None) -> CoverageIterationSettings:
    values = dict(os.environ if environ is None else environ)
    return CoverageIterationSettings(
        max_iterations_sufficient_coverage=_env_int(_ENV_BY_KEY["max_iterations_sufficient_coverage"], values),
        max_iterations_probably_sufficient_coverage=_env_int(
            _ENV_BY_KEY["max_iterations_probably_sufficient_coverage"],
            values,
        ),
        web_search_iteration_bonus=_env_int(_ENV_BY_KEY["web_search_iteration_bonus"], values),
    )


def _clamp(key: str, value: int) -> int:
    low, high = _BOUNDS[key]
    return max(low, min(high, value))


def resolve_coverage_iteration_settings(
    overrides: Mapping[str, Any] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> CoverageIterationSettings:
    base = load_coverage_iteration_settings(environ)
    merged = base.to_dict()
    if isinstance(overrides, Mapping):
        for key in _COVERAGE_KEYS:
            if key not in overrides or overrides[key] is None:
                continue
            merged[key] = _clamp(key, _parse_int(overrides[key], name=key))
    return CoverageIterationSettings(**merged)


def merge_coverage_iteration_overrides(
    stored: Mapping[str, Any] | None,
    updates: Mapping[str, Any] | None,
) -> dict[str, int]:
    """Merge partial persisted overrides; omitted keys are dropped from storage."""
    prior = dict(stored) if isinstance(stored, Mapping) else {}
    if not isinstance(updates, Mapping):
        return {key: prior[key] for key in _COVERAGE_KEYS if key in prior}
    for key in _COVERAGE_KEYS:
        if key not in updates:
            continue
        value = updates[key]
        if value is None:
            prior.pop(key, None)
        else:
            prior[key] = _clamp(key, _parse_int(value, name=key))
    return prior


def coverage_iteration_field_specs() -> list[dict[str, Any]]:
    """UI/API metadata derived from env-backed defaults."""
    defaults = load_coverage_iteration_settings()
    specs: list[dict[str, Any]] = []
    for key in _COVERAGE_KEYS:
        low, high = _BOUNDS[key]
        specs.append(
            {
                "key": key,
                "type": "integer",
                "minimum": low,
                "maximum": high,
                "default": defaults.to_dict()[key],
                "env_var": _ENV_BY_KEY[key],
            }
        )
    return specs