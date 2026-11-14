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
    return options


def thread_settings_baseline(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Subset applied via PUT /threads/{id}/settings after workflow is set."""
    cfg = config if config is not None else load_benchmark_config()
    baseline = cfg.get("thread_settings")
    return dict(baseline) if isinstance(baseline, dict) else {}


def context_baseline_record(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Snapshot stored on benchmark reports."""
    cfg = config if config is not None else load_benchmark_config()
    return {
        "config_path": cfg.get("_config_path", str(DEFAULT_CONFIG)),
        "context": chat_request_options(cfg),
        "thread_settings": thread_settings_baseline(cfg),
    }