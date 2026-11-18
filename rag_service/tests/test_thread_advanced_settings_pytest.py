from runtime_protocol.context_budget import compute_prefetch_budget
from runtime_protocol.thread_advanced_settings import (
    merge_context_budget_overrides,
    merge_thread_advanced_storage,
    resolve_thread_advanced_settings,
)
from app.models.llm_server_client import merge_thread_settings


def _ratio_env() -> dict[str, str]:
    return {
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
        "DEFAULT_TOKEN_BUDGET": "8192",
        "MAX_ITERATIONS_SUFFICIENT_COVERAGE": "2",
        "MAX_ITERATIONS_PROBABLY_SUFFICIENT_COVERAGE": "4",
        "WEB_SEARCH_ITERATION_BONUS": "1",
    }


def test_thread_override_changes_prefetch_budget():
    resolved = resolve_thread_advanced_settings(
        {"context_budget": {"recent_ratio": 0.40}},
        environ=_ratio_env(),
    )
    baseline = resolve_thread_advanced_settings({}, environ=_ratio_env())
    assert resolved["context_budget"]["recent_ratio"] == 0.40
    assert resolved["context_budget"]["recent_ratio"] != baseline["context_budget"]["recent_ratio"]
    base_budget = compute_prefetch_budget(8192, baseline["context_budget"])
    overridden = compute_prefetch_budget(8192, resolved["context_budget"])
    assert overridden["recent_history_chars"] > base_budget["recent_history_chars"]


def test_reset_context_budget_restores_env_default():
    stored = merge_context_budget_overrides({"recent_ratio": 0.40}, {"recent_ratio": None})
    assert "recent_ratio" not in stored
    resolved = resolve_thread_advanced_settings({"context_budget": stored}, environ=_ratio_env())
    assert resolved["context_budget"]["recent_ratio"] == 0.22


def test_merge_thread_settings_applies_context_window_override():
    merged = merge_thread_settings({"context_window": 131072})
    assert merged["context_window"] == 131072


def test_storage_reset_drops_context_window_override():
    stored = merge_thread_advanced_storage({"context_window": 200_000}, {"context_window": None})
    assert "context_window" not in stored