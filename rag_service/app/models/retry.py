from __future__ import annotations

import inspect
import logging
import re
from typing import Any, Callable, Dict, Optional

from app.services.retry import run_with_bounded_retries


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
    def retry_model_error(exc: BaseException) -> bool:
        retryable, _ = is_retryable_model_error(str(exc))
        return retryable

    async def observe_model_retry(event: Dict[str, Any]) -> None:
        if retry_observer is None:
            return
        message = str(event.get("exception_message") or "")
        _, reason = is_retryable_model_error(message)
        enriched = {
            **event,
            "reason": reason,
            "http_status_code": _extract_http_status_code(message),
        }
        observed = retry_observer(enriched)
        if inspect.isawaitable(observed):
            await observed

    return await run_with_bounded_retries(
        lambda: func(*args, **kwargs),
        max_attempts=10,
        base_delay_seconds=2,
        max_delay_seconds=32,
        retry_if=retry_model_error,
        operation_name="model call",
        retry_observer=observe_model_retry if retry_observer is not None else None,
    )
