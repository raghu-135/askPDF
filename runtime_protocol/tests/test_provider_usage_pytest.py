import pytest

from runtime_protocol.provider_usage import provider_cost_usd, usage_with_provider_cost


def test_provider_cost_usd_reads_openrouter_cost():
    assert provider_cost_usd({"cost": 0.0012, "total_tokens": 99}) == pytest.approx(0.0012)


def test_provider_cost_usd_ignores_missing_and_invalid_values():
    assert provider_cost_usd(None) is None
    assert provider_cost_usd({"total_tokens": 42}) is None
    assert provider_cost_usd({"cost": "n/a"}) is None
    assert provider_cost_usd({"cost": -1}) is None


def test_usage_with_provider_cost_merges_metadata_usage():
    merged = usage_with_provider_cost(
        {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        response_metadata={"token_usage": {"cost": 0.0042}},
    )
    assert merged["cost"] == pytest.approx(0.0042)
    assert merged["total_tokens"] == 5


def test_usage_with_provider_cost_never_estimates_from_tokens():
    merged = usage_with_provider_cost({"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000})
    assert "cost" not in merged
