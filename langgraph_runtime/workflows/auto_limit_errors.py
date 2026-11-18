"""Auto orchestrator limit exceptions."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from runtime_protocol.auto_limits import auto_limit_agent_error, auto_limit_chat_message


class AutoLimitExceeded(RuntimeError):
    """Raised when a thread Auto limit blocks orchestration."""

    def __init__(
        self,
        kind: str,
        *,
        limit: int,
        state: Mapping[str, Any] | None = None,
    ) -> None:
        self.kind = kind
        self.limit = limit
        self.agent_workflow_state: dict[str, Any] = dict(state) if isinstance(state, Mapping) else {}
        super().__init__(auto_limit_chat_message(kind, limit=limit))

    @property
    def user_message(self) -> str:
        return str(self)

    def as_agent_error(self) -> dict[str, object]:
        return auto_limit_agent_error(self.kind, limit=self.limit)


def workflow_budget_as_auto_limit(exc: Any, state: Mapping[str, Any] | None) -> AutoLimitExceeded | None:
    """Map visit-budget exhaustion to Auto limit messaging when the run is Auto."""

    workflow_id = str((state or {}).get("workflow_id") or "")
    if workflow_id != "auto_rag_agent":
        return None
    policy = (state or {}).get("loop_policy")
    if not isinstance(policy, dict):
        return None
    try:
        configured = int(policy.get("max_total_visits"))
    except (TypeError, ValueError):
        return None
    limit = int(getattr(exc, "limit", configured) or configured)
    if limit != configured:
        return None
    merged: MutableMapping[str, Any] = dict(state or {})
    if hasattr(exc, "as_dict"):
        merged["workflow_budget"] = exc.as_dict()
    return AutoLimitExceeded("max_total_visits", limit=limit, state=merged)