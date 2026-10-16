from langgraph_runtime.models.retry import is_retryable_model_error
from langgraph_runtime.workflows.runtime_invocation import error_summary, llm_result_metadata
from runtime_protocol.model_errors import (
    CONTEXT_LENGTH_EXCEEDED,
    classify_model_error_message,
    output_was_truncated,
    redact_provider_error_text,
    user_facing_model_error,
)


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


def test_redact_provider_error_text_strips_user_id():
    raw = 'Error: {"error":{"message":"bad","user_id":"secret-123"}}'
    assert "secret-123" not in redact_provider_error_text(raw)
    assert "<redacted>" in redact_provider_error_text(raw)


def test_user_facing_model_error_uses_safe_message_without_raw_secrets():
    payload = user_facing_model_error('{"user_id":"leak"}', {"code": "provider_request_rejected", "retryable": False})
    assert payload["safe_message"]
    assert "leak" not in payload["safe_message"]


def test_finish_reason_length_sets_output_truncated():
    response = type("Resp", (), {
        "content": "partial",
        "usage_metadata": {},
        "response_metadata": {"finish_reason": "length"},
    })()
    meta = llm_result_metadata(response, model_name="test")
    assert meta["output_truncated"] is True
    assert output_was_truncated("max_tokens") is True
