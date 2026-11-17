"""User-facing Auto orchestrator limit messages (shared product + runtime)."""

from __future__ import annotations

AUTO_WORKFLOW_ID = "auto_rag_agent"

AUTO_LIMIT_MAX_DELEGATES_PER_TURN = "max_delegates_per_turn"
AUTO_LIMIT_MAX_TOTAL_VISITS = "max_total_visits"
AUTO_LIMIT_MAX_INTERRUPTS_PER_RUN = "max_interrupts_per_run"

_AUTO_LIMIT_THREAD_FIELD = {
    AUTO_LIMIT_MAX_DELEGATES_PER_TURN: "Max delegates / turn",
    AUTO_LIMIT_MAX_TOTAL_VISITS: "Max visits",
    AUTO_LIMIT_MAX_INTERRUPTS_PER_RUN: "Max interrupts",
}


def auto_limit_settings_label(kind: str) -> str:
    return _AUTO_LIMIT_THREAD_FIELD.get(kind, kind.replace("_", " "))


def auto_limit_chat_message(kind: str, *, limit: int) -> str:
    """Plain-language assistant message when an Auto thread limit blocks progress."""

    field = auto_limit_settings_label(kind)
    return (
        f"This Auto run hit the thread limit for **{field}** (cap {limit}). "
        f"Open **AI Prompt Settings** for this thread and raise **{field}** if you need more headroom."
    )


def auto_limit_agent_error(kind: str, *, limit: int) -> dict[str, object]:
    message = auto_limit_chat_message(kind, limit=limit)
    return {
        "code": "auto_limit_exceeded",
        "auto_limit_kind": kind,
        "limit": limit,
        "safe_message": message,
        "retryable": False,
        "partial": False,
    }