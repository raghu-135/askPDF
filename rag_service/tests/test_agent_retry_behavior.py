import pytest

from app.models import retry as retry_module
from app.services import retry as retry_service


def test_retry_classifier_does_not_retry_generic_bad_request_or_vendor_400():
    is_retryable, reason = retry_module.is_retryable_model_error(
        "Error code: 400 - {'error': 'Model was unloaded while the request was still in queue..'}".lower()
    )

    assert is_retryable is False
    assert reason == ""


@pytest.mark.parametrize("status_code", [408, 409, 429, 500, 502, 503, 504])
def test_retry_classifier_follows_openai_compatible_retry_status_codes(status_code):
    is_retryable, reason = retry_module.is_retryable_model_error(f"Error code: {status_code} - transient")

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

    monkeypatch.setattr(retry_service.asyncio, "sleep", fake_sleep)

    result = await retry_module.invoke_with_retry(flaky_call)

    assert result == "ok"
    assert calls == 2
    assert sleeps == [2]


@pytest.mark.asyncio
async def test_bounded_retry_retries_ddgs_timeout_and_reraises_after_bound(monkeypatch):
    calls = 0
    sleeps = []

    class TimeoutException(Exception):
        pass

    async def flaky_call():
        nonlocal calls
        calls += 1
        raise TimeoutException("provider timed out")

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(retry_service.asyncio, "sleep", fake_sleep)

    with pytest.raises(TimeoutException):
        await retry_service.run_with_bounded_retries(
            flaky_call,
            max_attempts=3,
            base_delay_seconds=0.5,
            max_delay_seconds=2,
        )

    assert calls == 3
    assert sleeps == [0.5, 1.0]


@pytest.mark.asyncio
async def test_bounded_retry_does_not_retry_non_transient_errors(monkeypatch):
    calls = 0

    async def invalid_call():
        nonlocal calls
        calls += 1
        raise ValueError("invalid query")

    with pytest.raises(ValueError, match="invalid query"):
        await retry_service.run_with_bounded_retries(invalid_call)

    assert calls == 1
