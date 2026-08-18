from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from app.agent_workflows.enums import (
    AgentRunResumeAction,
    HitlSelectionMode,
    InterruptStatus,
    RESUME_ACTIONS,
    TERMINAL_INTERRUPT_STATUSES,
)
from app.db.models_sqlmodel import AgentRun
from app.time_utils import iso_utc_z, parse_datetime_utc, utc_now


INTERRUPT_STATUS_PENDING = InterruptStatus.PENDING.value
INTERRUPT_STATUS_RESUMED = InterruptStatus.RESUMED.value
INTERRUPT_STATUS_REJECTED = InterruptStatus.REJECTED.value
INTERRUPT_STATUS_EXPIRED = InterruptStatus.EXPIRED.value

PENDING_INTERRUPT_MAX_BYTES = 16_000
PENDING_INTERRUPT_STRING_LIMIT = 2_000
PENDING_INTERRUPT_LIST_LIMIT = 20
PENDING_INTERRUPT_DICT_LIMIT = 50
SUPPORTED_SPEC_SCHEMA_VERSION = 2
INTERRUPT_RESUME_GUARD_SCHEMA_VERSION = 1


class AgentRunInterruptError(ValueError):
    """Raised when an interrupt transition request is invalid."""

    def __init__(self, code: str, message: str, *, http_status: int = 409):
        super().__init__(message)
        self.code = code
        self.http_status = http_status


@dataclass
class InterruptResolutionResult:
    run: AgentRun
    outcome: str
    interrupt: Dict[str, Any]
    duplicate: bool = False


def compact_interrupt_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[truncated]"
    if isinstance(value, str):
        text = " ".join(value.split())
        if len(text) <= PENDING_INTERRUPT_STRING_LIMIT:
            return text
        return text[:PENDING_INTERRUPT_STRING_LIMIT].rstrip() + "..."
    if isinstance(value, list):
        return [compact_interrupt_value(item, depth=depth + 1) for item in value[:PENDING_INTERRUPT_LIST_LIMIT]]
    if isinstance(value, dict):
        compacted: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= PENDING_INTERRUPT_DICT_LIMIT:
                compacted["_truncated"] = True
                break
            compacted[str(key)] = compact_interrupt_value(item, depth=depth + 1)
        return compacted
    return value


def normalize_pending_interrupt_payload(payload: Dict[str, Any], *, requested_at: Optional[datetime] = None) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("pending interrupt payload must be an object")

    now = requested_at or utc_now()
    normalized = compact_interrupt_value(dict(payload))
    if not isinstance(normalized, dict):
        normalized = {}

    interrupt_id = str(normalized.get("interrupt_id") or uuid.uuid4())
    normalized["interrupt_id"] = interrupt_id
    normalized["status"] = str(normalized.get("status") or INTERRUPT_STATUS_PENDING)
    normalized["requested_at"] = str(normalized.get("requested_at") or iso_utc_z(now))

    allowed_actions = normalized.get("allowed_actions")
    if not isinstance(allowed_actions, list) or not all(isinstance(action, str) for action in allowed_actions):
        allowed_actions = [AgentRunResumeAction.APPROVE.value, AgentRunResumeAction.REJECT.value]
    normalized["allowed_actions"] = allowed_actions

    resume_version = normalized.get("resume_version")
    try:
        resume_version = int(resume_version)
    except (TypeError, ValueError):
        resume_version = 1
    normalized["resume_version"] = max(1, resume_version)

    encoded = json.dumps(normalized, ensure_ascii=True, sort_keys=True, default=str)
    if len(encoded.encode("utf-8")) > PENDING_INTERRUPT_MAX_BYTES:
        raise ValueError("pending interrupt payload is too large")
    return normalized


def pending_interrupt_from_run(run: AgentRun) -> Dict[str, Any]:
    pending = run.pending_interrupt_json if isinstance(run.pending_interrupt_json, dict) else {}
    return dict(pending)


def interrupt_expired(interrupt: Dict[str, Any], now: datetime) -> bool:
    expires_at = parse_datetime_utc(interrupt.get("expires_at"))
    return bool(expires_at and expires_at <= now)


def terminal_decision_matches(interrupt: Dict[str, Any], *, action: str, interrupt_id: str) -> bool:
    decision = interrupt.get("decision")
    if not isinstance(decision, dict):
        return False
    if decision.get("interrupt_id") != interrupt_id:
        return False
    return decision.get("action") == action or decision.get("requested_action") == action


def option_ids(interrupt: Dict[str, Any]) -> set[str]:
    options = interrupt.get("options") if isinstance(interrupt.get("options"), list) else []
    return {
        str(option.get("id"))
        for option in options
        if isinstance(option, dict) and isinstance(option.get("id"), str) and option.get("id")
    }


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value if isinstance(value, dict) else {}, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_interrupt_resume_guard(run: AgentRun) -> Dict[str, Any]:
    spec = run.resolved_spec_json if isinstance(run.resolved_spec_json, dict) else {}
    return {
        "schema_version": INTERRUPT_RESUME_GUARD_SCHEMA_VERSION,
        "spec_schema_version": spec.get("schema_version"),
        "workflow_id": run.workflow_id,
        "workflow_version_id": run.workflow_version_id,
        "workflow_version": run.workflow_version,
        "checkpoint_thread_id": run.checkpoint_thread_id,
        "resolved_spec_hash": canonical_json_hash(spec),
    }


def validate_interrupt_resume_guard(interrupt: Dict[str, Any], run: AgentRun) -> None:
    resume_guard = interrupt.get("resume_guard") if isinstance(interrupt.get("resume_guard"), dict) else {}
    if not resume_guard:
        raise AgentRunInterruptError(
            "interrupt_resume_guard_missing",
            "The pending interrupt is missing run resume guard metadata.",
        )

    expected = run_interrupt_resume_guard(run)
    if resume_guard.get("spec_schema_version") != SUPPORTED_SPEC_SCHEMA_VERSION:
        raise AgentRunInterruptError(
            "interrupt_spec_schema_unsupported",
            "The pending interrupt was created for an unsupported agent workflow schema.",
        )

    fields = (
        "schema_version",
        "spec_schema_version",
        "workflow_id",
        "checkpoint_thread_id",
        "resolved_spec_hash",
    )
    mismatched = [field for field in fields if resume_guard.get(field) != expected.get(field)]
    if mismatched:
        raise AgentRunInterruptError(
            "interrupt_resume_guard_mismatch",
            "The pending interrupt resume guard no longer matches the stored agent run.",
        )


def build_interrupt_decision(
    *,
    interrupt_id: str,
    action: str,
    decided_at: datetime,
    edited_payload: Optional[Dict[str, Any]],
    client_metadata: Optional[Dict[str, Any]],
    resume_version: Optional[int],
    selected_option_ids: Optional[list[str]],
) -> Dict[str, Any]:
    decision = {
        "interrupt_id": interrupt_id,
        "action": action,
        "decided_at": iso_utc_z(decided_at),
    }
    if resume_version is not None:
        decision["resume_version"] = resume_version
    if edited_payload is not None:
        decision["edited_payload"] = compact_interrupt_value(edited_payload)
    if selected_option_ids is not None:
        decision["selected_option_ids"] = compact_interrupt_value(selected_option_ids)
    if client_metadata is not None:
        decision["client_metadata"] = compact_interrupt_value(client_metadata)
    return decision

def validate_pending_interrupt_request(
    interrupt: Dict[str, Any],
    *,
    interrupt_id: str,
    action: str,
    resume_token: Optional[str],
    resume_version: Optional[int],
    selected_option_ids: Optional[list[str]],
) -> None:
    if interrupt.get("interrupt_id") != interrupt_id:
        raise AgentRunInterruptError(
            "interrupt_mismatch",
            "The requested interrupt does not match the run's current interrupt.",
        )

    allowed_actions = interrupt.get("allowed_actions") if isinstance(interrupt.get("allowed_actions"), list) else []
    if action not in allowed_actions:
        raise AgentRunInterruptError(
            "interrupt_action_not_allowed",
            f"Action {action!r} is not allowed for this interrupt.",
            http_status=400,
        )

    expected_token = interrupt.get("resume_token")
    if resume_token is not None and expected_token is not None and resume_token != expected_token:
        raise AgentRunInterruptError(
            "resume_token_mismatch",
            "The resume token does not match the current interrupt.",
        )

    if resume_version is not None and int(interrupt.get("resume_version") or 1) != resume_version:
        raise AgentRunInterruptError(
            "resume_version_mismatch",
            "The resume version does not match the current interrupt.",
        )

    if action == AgentRunResumeAction.APPROVE_SELECTED.value:
        valid_option_ids = option_ids(interrupt)
        if not valid_option_ids:
            raise AgentRunInterruptError(
                "interrupt_options_missing",
                "This interrupt does not expose selectable options.",
                http_status=400,
            )
        if not selected_option_ids:
            raise AgentRunInterruptError(
                "interrupt_selection_required",
                "At least one option must be selected.",
                http_status=400,
            )
        invalid_option_ids = sorted(set(map(str, selected_option_ids)) - valid_option_ids)
        if invalid_option_ids:
            raise AgentRunInterruptError(
                "interrupt_selection_invalid",
                f"Selected option ids are invalid: {', '.join(invalid_option_ids)}",
                http_status=400,
            )
        selection_mode = str(interrupt.get("selection_mode") or HitlSelectionMode.SINGLE.value)
        if selection_mode == HitlSelectionMode.SINGLE.value and len(selected_option_ids) != 1:
            raise AgentRunInterruptError(
                "interrupt_selection_count_invalid",
                "This interrupt requires exactly one selected option.",
                http_status=400,
            )
