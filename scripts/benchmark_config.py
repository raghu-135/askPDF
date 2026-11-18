"""Load pinned benchmark run configuration (context window, packing ratios, thread defaults)."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "benchmarks" / "benchmark_config.json"

# Keys accepted by ThreadChatRequest.context_budget (ContextBudgetOverride).
_CHAT_BUDGET_KEYS = frozenset(
    {
        "usable_ratio",
        "recent_ratio",
        "semantic_ratio",
        "durable_memory_ratio",
        "document_ratio",
        "memory_expanded_ratio",
        "evidence_ratio",
        "draft_ratio",
        "plan_section_ratio",
        "plan_total_ratio",
        "plan_memory_ratio",
    }
)

_THREAD_ADVANCED_KEYS = frozenset(
    {
        "context_window",
        "context_budget",
        "max_iterations_sufficient_coverage",
        "max_iterations_probably_sufficient_coverage",
        "web_search_iteration_bonus",
        "coverage_iterations",
    }
)


def _config_path() -> Path:
    raw = os.getenv("ASKPDF_BENCHMARK_CONFIG", "").strip()
    return Path(raw) if raw else DEFAULT_CONFIG


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_benchmark_config() -> dict[str, Any]:
    path = _config_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = os.getenv("ASKPDF_BENCHMARK_CONTEXT_OVERRIDES_JSON", "").strip()
    if raw:
        data = _deep_merge(data, json.loads(raw))
    try:
        data["_config_path"] = str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        data["_config_path"] = str(path)
    return data


def chat_request_options(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Fields merged into POST /threads/{id}/chat (no per-question variation in baseline)."""
    cfg = config if config is not None else load_benchmark_config()
    ctx = cfg.get("context") if isinstance(cfg.get("context"), dict) else {}
    options: dict[str, Any] = {}
    for key in ("context_window", "use_web_search", "use_reranker"):
        if key in ctx:
            options[key] = ctx[key]
    budget = ctx.get("context_budget")
    if isinstance(budget, dict):
        options["context_budget"] = {k: budget[k] for k in _CHAT_BUDGET_KEYS if k in budget}
    advanced = cfg.get("advanced") if isinstance(cfg.get("advanced"), dict) else {}
    if advanced.get("context_window") is not None and "context_window" not in options:
        options["context_window"] = advanced["context_window"]
    adv_budget = advanced.get("context_budget")
    if isinstance(adv_budget, dict):
        merged = dict(options.get("context_budget") or {})
        merged.update({k: adv_budget[k] for k in _CHAT_BUDGET_KEYS if k in adv_budget})
        options["context_budget"] = merged
    return options


def thread_settings_baseline(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Subset applied via PUT /threads/{id}/settings after workflow is set."""
    cfg = config if config is not None else load_benchmark_config()
    baseline = cfg.get("thread_settings")
    if not isinstance(baseline, dict):
        return {}
    advanced = cfg.get("advanced") if isinstance(cfg.get("advanced"), dict) else {}
    merged = dict(baseline)
    for key in _THREAD_ADVANCED_KEYS:
        if key in advanced:
            merged[key] = advanced[key]
    return merged


def context_baseline_record(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Snapshot stored on benchmark reports."""
    cfg = config if config is not None else load_benchmark_config()
    return {
        "config_path": cfg.get("_config_path", str(DEFAULT_CONFIG)),
        "context": chat_request_options(cfg),
        "thread_settings": thread_settings_baseline(cfg),
    }