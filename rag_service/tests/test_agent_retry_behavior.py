import pytest

from app.agent import agent as agent_module


def test_retry_classifier_does_not_retry_generic_bad_request_or_vendor_400():
    is_retryable, reason = agent_module._is_retryable_model_error(
        "Error code: 400 - {'error': 'Model was unloaded while the request was still in queue..'}".lower()
    )

    assert is_retryable is False
    assert reason == ""


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 502, 503, 504])
def test_retry_classifier_follows_openai_compatible_retry_status_codes(status_code):
    is_retryable, reason = agent_module._is_retryable_model_error(f"Error code: {status_code} - transient")

    assert is_retryable is True
    assert reason == f"Retryable OpenAI-compatible API error ({status_code})"


@pytest.mark.asyncio
async def test_invoke_with_retry_retries_openai_compatible_status(monkeypatch):
    calls = 0
    sleeps = []

    async def flaky_call():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Exception("Error code: 503 - {'error': 'temporarily unavailable'}")
        return "ok"

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(agent_module.asyncio, "sleep", fake_sleep)

    result = await agent_module.invoke_with_retry(flaky_call)

    assert result == "ok"
    assert calls == 2
    assert sleeps == [2]
