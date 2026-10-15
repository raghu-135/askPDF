import os

import pytest

from runtime_protocol.context_budget import (
    compute_prefetch_budget,
    load_context_budget_settings,
    resolve_context_budget,
    usable_context_chars,
)
from runtime_protocol.configuration import validate_runtime_environment


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
    }


def test_prefetch_budget_follows_env_ratios():
    settings = load_context_budget_settings(_ratio_env())
    budget = compute_prefetch_budget(8192, settings)
    usable = int(8192 * settings.usable_ratio * 4)
    assert budget["recent_history_chars"] == int(usable * settings.recent_ratio)
    assert budget["document_context_chars"] == int(usable * settings.document_ratio)


def test_request_override_replaces_prefetch_ratios():
    settings = resolve_context_budget({"recent_ratio": 0.10}, environ=_ratio_env())
    assert settings.recent_ratio == 0.10
    assert settings.semantic_ratio == 0.12


def test_prefetch_sum_above_one_is_rejected():
    with pytest.raises(ValueError, match="sum to at most 1"):
        resolve_context_budget(
            {
                "recent_ratio": 0.40,
                "semantic_ratio": 0.40,
                "durable_memory_ratio": 0.40,
                "document_ratio": 0.40,
            },
            environ=_ratio_env(),
        )


def test_usable_context_chars_token_formula():
    settings = load_context_budget_settings(_ratio_env())
    window = 10_000
    expected = int(window * settings.usable_ratio * 4 * settings.recent_ratio)
    assert usable_context_chars(window, ratio_key="recent_ratio", budget=settings) == expected


def test_startup_rejects_missing_context_ratio():
    from rag_service.tests.test_runtime_configuration_pytest import _environment
    from runtime_protocol.configuration import RuntimeConfigurationError

    env = _environment()
    env.pop("CONTEXT_RECENT_RATIO", None)
    with pytest.raises(RuntimeConfigurationError):
        validate_runtime_environment(service="control_plane", environ={**env, "COMPOSE_PROFILES": ""})
