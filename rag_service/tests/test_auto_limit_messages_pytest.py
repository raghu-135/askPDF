from __future__ import annotations

from runtime_protocol.auto_limits import (
    AUTO_LIMIT_MAX_DELEGATES_PER_TURN,
    AUTO_LIMIT_MAX_TOTAL_VISITS,
    auto_limit_agent_error,
    auto_limit_chat_message,
)


def test_auto_limit_chat_message_mentions_thread_settings():
    message = auto_limit_chat_message(AUTO_LIMIT_MAX_DELEGATES_PER_TURN, limit=1)
    assert "AI Prompt Settings" in message
    assert "Max delegates" in message
    assert "1" in message


def test_auto_limit_agent_error_shape():
    error = auto_limit_agent_error(AUTO_LIMIT_MAX_TOTAL_VISITS, limit=12)
    assert error["code"] == "auto_limit_exceeded"
    assert error["auto_limit_kind"] == AUTO_LIMIT_MAX_TOTAL_VISITS
    assert error["limit"] == 12
    assert "safe_message" in error
    assert error["retryable"] is False