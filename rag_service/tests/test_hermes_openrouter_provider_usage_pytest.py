from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_runtime.hermes_pinned_patch import openrouter_provider_usage as or_usage


@pytest.mark.parametrize(
    ("base_url", "model", "provider"),
    [
        ("https://openrouter.ai/api/v1", "openai/gpt-4.1-nano", "custom"),
        ("http://host.docker.internal:12345/v1", "openai/gpt-4.1-nano", "custom"),
        ("http://lmstudio:1234/v1", "openai/gpt-4.1-nano", "openrouter"),
    ],
)
def test_openrouter_usage_accounting_enabled_for_proxy_and_direct(base_url, model, provider):
    assert or_usage.openrouter_usage_accounting_enabled(base_url=base_url, model=model, provider=provider)


def test_openrouter_usage_accounting_disabled_for_plain_local_model():
    assert not or_usage.openrouter_usage_accounting_enabled(
        base_url="http://127.0.0.1:1234/v1",
        model="local-model",
        provider="custom",
    )


def test_merge_openrouter_usage_request_sets_include_and_stream_options():
    merged = or_usage.merge_openrouter_usage_request(
        {"model": "openai/gpt-4.1-nano", "stream": True},
        base_url="http://host.docker.internal:12345/v1",
        model="openai/gpt-4.1-nano",
        provider="custom",
    )
    assert merged["extra_body"]["usage"] == {"include": True}
    assert merged["stream_options"] == {"include_usage": True}
    assert "usage" not in merged


def test_merge_openrouter_usage_request_respects_existing_stream_options():
    merged = or_usage.merge_openrouter_usage_request(
        {
            "model": "openai/gpt-4.1-nano",
            "stream": True,
            "stream_options": {"include_usage": False},
        },
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-4.1-nano",
        provider="custom",
    )
    assert merged["stream_options"]["include_usage"] is True


def test_provider_cost_from_response_usage_reads_openrouter_cost():
    assert or_usage.provider_cost_from_response_usage({"cost": 0.0025, "total_tokens": 30}) == pytest.approx(0.0025)


def test_accumulate_agent_provider_cost_sums_session_total():
    agent = SimpleNamespace()
    or_usage.accumulate_agent_provider_cost(agent, SimpleNamespace(usage={"cost": 0.001}))
    or_usage.accumulate_agent_provider_cost(agent, SimpleNamespace(usage={"cost": 0.0004}))
    assert agent._askpdf_provider_cost_usd == pytest.approx(0.0014)


def test_enrich_terminal_usage_adds_cost_without_estimating_from_tokens():
    agent = SimpleNamespace(_askpdf_provider_cost_usd=0.0033)
    enriched = or_usage.enrich_terminal_usage({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}, agent)
    assert enriched["cost"] == pytest.approx(0.0033)
    assert enriched["provider_cost_usd"] == pytest.approx(0.0033)
