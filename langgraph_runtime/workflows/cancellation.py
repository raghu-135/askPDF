"""Cooperative cancellation backed by runtime-local execution-store probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping


@dataclass(frozen=True)
class ChatRunCancelResult:
    status: str
    run_id: str | None = None
    run_status: str | None = None


class ChatRunCancellationRequested(Exception):
    def __init__(self, state: Mapping[str, Any] | None = None):
        super().__init__("Runtime execution cancellation requested")
        self.state = dict(state or {})


async def request_chat_run_cancel(run_id: str, *, thread_id: str) -> ChatRunCancelResult:
    del thread_id
    return ChatRunCancelResult(status="cancel_requested", run_id=run_id, run_status="running")


async def raise_if_chat_run_cancelled(
    checker: Callable[[], Awaitable[bool]] | None,
    state: Mapping[str, Any] | None = None,
) -> None:
    if checker is not None and await checker():
        raise ChatRunCancellationRequested(state)
