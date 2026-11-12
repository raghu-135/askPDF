import pytest

from app.product_orchestration.trace_otel import _llm_usage_metrics
from app.services.agent_task_runtime_projection import _merge_budget
from app.services.agent_task_budgets import initial_budget_state
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from runtime_protocol.provider_usage import provider_cost_usd, usage_with_provider_cost


def test_provider_cost_usd_rejects_token_only_payloads():
    assert provider_cost_usd({"total_tokens": 500}) is None


def test_usage_with_provider_cost_reads_openrouter_metadata():
    merged = usage_with_provider_cost(
        {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        response_metadata={"token_usage": {"cost": 0.0025}},
    )
    assert merged["total_tokens"] == 20
    assert merged["cost"] == pytest.approx(0.0025)


def test_merge_budget_hermes_flat_usage_infers_model_calls():
    limits = {"max_model_calls": 5, "max_model_tokens": 5000, "max_tool_calls": 5, "max_active_runtime_ms": 60_000}
    merged = _merge_budget(
        initial_budget_state(limits),
        {
            "operation_id": "op-1",
            "total_tokens": 128,
            "tool_calls": 2,
            "active_runtime_ms": 1500,
            "measured_dimensions": ["model_tokens", "tool_calls", "active_runtime_ms"],
        },
        limits,
    )
    assert merged["tranche_usage"]["model_tokens"] == 128
    assert merged["tranche_usage"]["model_calls"] == 1
    assert merged["tranche_usage"]["tool_calls"] == 2
    assert merged["tranche_usage"]["elapsed_active_ms"] == 1500


def test_llm_usage_metrics_sums_provider_cost():
    spans = [
        {
            "kind": OpenInferenceSpanKindValues.LLM.value,
            "attributes": {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 42,
                "llm.provider_cost_usd": 0.001,
            },
        },
        {
            "kind": OpenInferenceSpanKindValues.LLM.value,
            "attributes": {
                SpanAttributes.LLM_TOKEN_COUNT_TOTAL: 10,
                "llm.provider_cost_usd": 0.002,
            },
        },
    ]
    metrics = _llm_usage_metrics(spans)
    assert metrics["llm_provider_cost_usd"] == pytest.approx(0.003)
