from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from app.db import AgentRunStatus
from app.db.connection_sqlmodel import async_session_maker
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import AgentRun


CHAT_CANCEL_REQUESTED = "cancel_requested"
CHAT_CANCEL_ALREADY_TERMINAL = "already_terminal"
CHAT_CANCEL_AWAITING_HUMAN = "awaiting_human"
CHAT_CANCEL_UNSUPPORTED = "unsupported"
BUILDER_TEST_RUN_KIND = "builder_test"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatRunCancelResult:
    status: str
    run_id: Optional[str] = None
    run_status: Optional[str] = None


class ChatRunCancellationRequested(Exception):
    """Internal cooperative-cancellation signal carrying the latest graph state."""

    def __init__(self, state: Optional[Dict[str, Any]] = None):
        super().__init__("Chat run cancellation requested")
        self.state = dict(state or {})


async def request_chat_run_cancel(run_id: str, *, thread_id: str) -> ChatRunCancelResult:
    """Request cancellation for one ordinary running chat Agent Run."""

    async with async_session_maker() as session:
        async with session.begin():
            run = await session.get(AgentRun, run_id)
            if run is None or run.thread_id != thread_id:
                return ChatRunCancelResult(status="missing")
            metadata = dict(run.run_metadata_json or {})
            if metadata.get("run_kind") == BUILDER_TEST_RUN_KIND:
                return ChatRunCancelResult(
                    status=CHAT_CANCEL_UNSUPPORTED,
                    run_id=run.id,
                    run_status=run.status,
                )
            if run.status == AgentRunStatus.AWAITING_HUMAN.value:
                return ChatRunCancelResult(
                    status=CHAT_CANCEL_AWAITING_HUMAN,
                    run_id=run.id,
                    run_status=run.status,
                )
            if run.status != AgentRunStatus.RUNNING.value:
                return ChatRunCancelResult(
                    status=CHAT_CANCEL_ALREADY_TERMINAL,
                    run_id=run.id,
                    run_status=run.status,
                )
            metadata["cancel_requested"] = True
            replace_jsonb_field(run, "run_metadata_json", metadata)
            await session.flush()
            return ChatRunCancelResult(
                status=CHAT_CANCEL_REQUESTED,
                run_id=run.id,
                run_status=run.status,
            )


async def chat_run_cancel_requested(run_id: str) -> bool:
    """Read cancellation state in a fresh session so checks work across workers."""

    try:
        async with async_session_maker() as session:
            async with session.begin():
                run = await session.get(AgentRun, run_id)
                return bool(
                    run
                    and run.status == AgentRunStatus.RUNNING.value
                    and (run.run_metadata_json or {}).get("run_kind") != BUILDER_TEST_RUN_KIND
                    and (run.run_metadata_json or {}).get("cancel_requested")
                )
    except Exception:
        logger.warning(
            "Unable to read chat cancellation state; allowing the run to continue | run_id=%s",
            run_id,
            exc_info=True,
        )
        return False


async def raise_if_chat_run_cancelled(
    checker: Optional[Callable[[], Awaitable[bool]]],
    state: Optional[Dict[str, Any]] = None,
) -> None:
    if checker is not None and await checker():
        raise ChatRunCancellationRequested(state)
