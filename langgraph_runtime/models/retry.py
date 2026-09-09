from __future__ import annotations

import asyncio
import inspect
import logging
import re
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger(__name__)


def _extract_http_status_code(err_str: str) -> Optional[int]:
    patterns = (
        r"status(?:_code)?[=:]\s*(\d{3})",
        r"error code:\s*(\d{3})",
        r"\b(\d{3})\b",
    )
    for pattern in patterns:
        match = re.search(pattern, err_str)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def is_retryable_model_error(err_str: str) -> tuple[bool, str]:
    status_code = _extract_http_status_code(err_str)
    if status_code in {408, 409, 429} or (status_code is not None and status_code >= 500):
        return True, f"Retryable OpenAI-compatible API error ({status_code})"
    return False, ""


def _compact_error_message(value: Any, *, limit: int = 500) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


async def invoke_with_retry(
    func,
    *args,
    retry_observer: Optional[Callable[[Dict[str, Any]], Any]] = None,
    **kwargs,
):
    max_retries = 10
    base_delay = 2
    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            is_retryable, reason = is_retryable_model_error(err_str)
            if is_retryable:
                delay = base_delay * (2 ** min(i, 4))
                if retry_observer is not None:
                    event = {
                        "attempt": i + 1,
                        "delay_ms": delay * 1000,
                        "reason": reason,
                        "http_status_code": _extract_http_status_code(err_str),
                        "exception_type": type(e).__name__,
                        "exception_message": _compact_error_message(e),
                    }
                    observed = retry_observer(event)
                    if inspect.isawaitable(observed):
                        await observed
                logger.warning("%s. Retrying in %ss... (Attempt %s/%s)", reason, delay, i + 1, max_retries)
                await asyncio.sleep(delay)
                continue
            raise
    raise Exception("Max retries reached while waiting for model to become available.")
