from runtime_protocol.model_errors import (
    classify_model_error_message,
    redact_provider_error_text,
    user_facing_model_error,
)


def test_redact_provider_error_text_strips_user_id():
    raw = 'Error: {"error":{"message":"bad","user_id":"secret-123"}}'
    assert "secret-123" not in redact_provider_error_text(raw)
    assert "<redacted>" in redact_provider_error_text(raw)


def test_user_facing_model_error_uses_safe_message_without_raw_secrets():
    payload = user_facing_model_error('{"user_id":"leak"}', {"code": "provider_request_rejected", "retryable": False})
    assert payload["safe_message"]
    assert "leak" not in payload["safe_message"]


def test_classify_provider_rejection_without_leaking_user_id():
    classified = classify_model_error_message('Error 401: {"user_id":"acct-1"}')
    assert classified["code"] == "provider_request_rejected"
