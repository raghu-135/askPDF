"""Bounded retries for transient, side-effect-free service calls."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import httpx


logger = logging.getLogger(__name__)
T = TypeVar("T")


def is_transient_external_error(exc: BaseException) -> bool:
    """Return whether an external call may succeed if attempted again."""

    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError, TimeoutError, ConnectionError)):
        return True
    status_code = getattr(exc, "status_code", None) or getattr(exc, "response", None)
    if hasattr(status_code, "status_code"):
        status_code = status_code.status_code
    if isinstance(status_code, int):
        return status_code in {408, 409, 425, 429} or status_code >= 500
    # DDGS exposes provider timeouts as ddgs.exceptions.TimeoutException. Keep
    # this structural so the control plane does not require a DDGS import in
    # the shared retry policy.
    return exc.__class__.__name__ in {"TimeoutException", "ConnectError", "NetworkError"}


async def run_with_bounded_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_attempts: int = 3,
    base_delay_seconds: float = 1.0,
    max_delay_seconds: float = 8.0,
    retry_if: Callable[[BaseException], bool] = is_transient_external_error,
    operation_name: str = "external operation",
    retry_observer: Callable[[dict[str, Any]], Any] | None = None,
) -> T:
    """Run an async operation with bounded exponential backoff.

    ``max_attempts`` includes the initial call. The final exception is
    re-raised unchanged, preserving the provider-specific failure contract.
    """

    attempts = max(1, int(max_attempts))
    base_delay = max(0.0, float(base_delay_seconds))
    max_delay = max(base_delay, float(max_delay_seconds))
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if attempt >= attempts or not retry_if(exc):
                raise
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            event = {
                "attempt": attempt,
                "next_attempt": attempt + 1,
                "max_attempts": attempts,
                "delay_ms": round(delay * 1000),
                "exception_type": type(exc).__name__,
                "exception_message": " ".join(str(exc).split())[:500],
                "operation": operation_name,
            }
            if retry_observer is not None:
                observed = retry_observer(event)
                if inspect.isawaitable(observed):
                    await observed
            logger.warning(
                "%s failed transiently; retrying in %.2fs (attempt %s/%s): %s",
                operation_name,
                delay,
                attempt,
                attempts,
                type(exc).__name__,
            )
            await asyncio.sleep(delay)
    raise AssertionError("bounded retry loop exhausted without returning or raising")
