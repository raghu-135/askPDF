import pytest

from langgraph_runtime.models.retry import is_retryable_model_error
from langgraph_runtime.workflows.runtime_invocation import error_summary, llm_result_metadata
from runtime_protocol.model_errors import CONTEXT_LENGTH_EXCEEDED, classify_model_error_message, output_was_truncated

pytest.importorskip("langgraph")


def test_context_length_provider_error_is_non_retryable_and_coded():
    message = "Error code: 400 - {'error': {'code': 'context_length_exceeded'}}"
    classified = classify_model_error_message(message)
    assert classified["code"] == CONTEXT_LENGTH_EXCEEDED
    assert classified["retryable"] is False
    retryable, _ = is_retryable_model_error(message)
    assert retryable is False


def test_error_summary_uses_context_length_code():
    exc = Exception("maximum context length exceeded for this model")
    payload = error_summary(exc, code="router_llm_failed")
    assert payload["code"] == CONTEXT_LENGTH_EXCEEDED
    assert payload["retryable"] is False


def test_finish_reason_length_sets_output_truncated():
    response = type("Resp", (), {
        "content": "partial",
        "usage_metadata": {},
        "response_metadata": {"finish_reason": "length"},
    })()
    meta = llm_result_metadata(response, model_name="test")
    assert meta["output_truncated"] is True
    assert output_was_truncated("max_tokens") is True
