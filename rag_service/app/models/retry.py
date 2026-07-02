from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional


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


async def invoke_with_retry(func, *args, **kwargs):
    max_retries = 10
    base_delay = 2
    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_retryable, reason = is_retryable_model_error(err_str)
            if is_retryable:
                delay = base_delay * (2 ** min(i, 4))
                logger.warning("%s. Retrying in %ss... (Attempt %s/%s)", reason, delay, i + 1, max_retries)
                await asyncio.sleep(delay)
                continue
            raise
    raise Exception("Max retries reached while waiting for model to become available.")
