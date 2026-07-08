from __future__ import annotations

import json
import hashlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.agent_patterns.checkpointing import delete_agent_checkpoints
from app.agent_patterns.debug_trace import append_interrupt_event_to_debug_payload, append_runtime_event_to_debug_payload
from app.agent_patterns.builtin_workflows import builtin_workflow_keys, load_builtin_workflows
from app.agent_patterns.validator import TemplateValidationError, TemplateValidator
from app.db.connection_sqlmodel import async_session_maker
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import (
    AgentWorkflow,
    AgentRun,
    ChatTurn,
)
from app.time_utils import iso_utc_z, parse_datetime_utc, utc_now


RUN_STATUS_RUNNING = "running"
RUN_STATUS_AWAITING_HUMAN = "awaiting_human"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_CLARIFICATION = "clarification"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_REJECTED = "rejected"
RUN_STATUS_EXPIRED = "expired"
CHECKPOINT_PRUNABLE_RUN_STATUSES = {
    RUN_STATUS_COMPLETED,
    RUN_STATUS_CLARIFICATION,
    RUN_STATUS_FAILED,
    RUN_STATUS_REJECTED,
    RUN_STATUS_EXPIRED,
}

INTERRUPT_STATUS_PENDING = "pending"
INTERRUPT_STATUS_RESUMED = "resumed"
INTERRUPT_STATUS_REJECTED = "rejected"
INTERRUPT_STATUS_EXPIRED = "expired"

RESUME_ACTIONS = {"approve", "approve_selected", "edit", "continue_without"}
TERMINAL_INTERRUPT_STATUSES = {
    INTERRUPT_STATUS_RESUMED,
    INTERRUPT_STATUS_REJECTED,
    INTERRUPT_STATUS_EXPIRED,
}

PENDING_INTERRUPT_MAX_BYTES = 16_000
PENDING_INTERRUPT_STRING_LIMIT = 2_000
PENDING_INTERRUPT_LIST_LIMIT = 20
PENDING_INTERRUPT_DICT_LIMIT = 50
SUPPORTED_SPEC_SCHEMA_VERSION = 2
INTERRUPT_COMPATIBILITY_SCHEMA_VERSION = 1


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


def _compact_interrupt_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "[truncated]"
    if isinstance(value, str):
        text = " ".join(value.split())
        if len(text) <= PENDING_INTERRUPT_STRING_LIMIT:
            return text
        return text[:PENDING_INTERRUPT_STRING_LIMIT].rstrip() + "..."
    if isinstance(value, list):
        return [_compact_interrupt_value(item, depth=depth + 1) for item in value[:PENDING_INTERRUPT_LIST_LIMIT]]
    if isinstance(value, dict):
        compacted: Dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= PENDING_INTERRUPT_DICT_LIMIT:
                compacted["_truncated"] = True
                break
            compacted[str(key)] = _compact_interrupt_value(item, depth=depth + 1)
        return compacted
    return value


def normalize_pending_interrupt_payload(payload: Dict[str, Any], *, requested_at: Optional[datetime] = None) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("pending interrupt payload must be an object")

    now = requested_at or utc_now()
    normalized = _compact_interrupt_value(dict(payload))
    if not isinstance(normalized, dict):
        normalized = {}

    interrupt_id = str(normalized.get("interrupt_id") or uuid.uuid4())
    normalized["interrupt_id"] = interrupt_id
    normalized["status"] = str(normalized.get("status") or INTERRUPT_STATUS_PENDING)
    normalized["requested_at"] = str(normalized.get("requested_at") or iso_utc_z(now))

    allowed_actions = normalized.get("allowed_actions")
    if not isinstance(allowed_actions, list) or not all(isinstance(action, str) for action in allowed_actions):
        allowed_actions = ["approve", "reject"]
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


def _pending_interrupt_from_run(run: AgentRun) -> Dict[str, Any]:
    pending = run.pending_interrupt_json if isinstance(run.pending_interrupt_json, dict) else {}
    return dict(pending)


def _interrupt_expired(interrupt: Dict[str, Any], now: datetime) -> bool:
    expires_at = parse_datetime_utc(interrupt.get("expires_at"))
    return bool(expires_at and expires_at <= now)


def _terminal_decision_matches(interrupt: Dict[str, Any], *, action: str, interrupt_id: str) -> bool:
    decision = interrupt.get("decision")
    if not isinstance(decision, dict):
        return False
    if decision.get("interrupt_id") != interrupt_id:
        return False
    return decision.get("action") == action or decision.get("requested_action") == action


def _option_ids(interrupt: Dict[str, Any]) -> set[str]:
    options = interrupt.get("options") if isinstance(interrupt.get("options"), list) else []
    return {
        str(option.get("id"))
        for option in options
        if isinstance(option, dict) and isinstance(option.get("id"), str) and option.get("id")
    }


def _canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value if isinstance(value, dict) else {}, ensure_ascii=True, sort_keys=True, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _run_interrupt_compatibility(run: AgentRun) -> Dict[str, Any]:
    spec = run.resolved_spec_json if isinstance(run.resolved_spec_json, dict) else {}
    return {
        "schema_version": INTERRUPT_COMPATIBILITY_SCHEMA_VERSION,
        "spec_schema_version": spec.get("schema_version"),
        "workflow_id": run.workflow_id,
        "checkpoint_thread_id": run.checkpoint_thread_id,
        "resolved_spec_hash": _canonical_json_hash(spec),
    }


def _validate_interrupt_compatibility(interrupt: Dict[str, Any], run: AgentRun) -> None:
    compatibility = interrupt.get("compatibility") if isinstance(interrupt.get("compatibility"), dict) else {}
    if not compatibility:
        raise AgentRunInterruptError(
            "interrupt_compatibility_missing",
            "The pending interrupt is missing run compatibility metadata.",
        )

    expected = _run_interrupt_compatibility(run)
    if compatibility.get("spec_schema_version") != SUPPORTED_SPEC_SCHEMA_VERSION:
        raise AgentRunInterruptError(
            "interrupt_spec_schema_unsupported",
            "The pending interrupt was created for an unsupported agent pattern schema.",
        )

    fields = (
        "schema_version",
        "spec_schema_version",
        "workflow_id",
        "checkpoint_thread_id",
        "resolved_spec_hash",
    )
    mismatched = [field for field in fields if compatibility.get(field) != expected.get(field)]
    if mismatched:
        raise AgentRunInterruptError(
            "interrupt_compatibility_mismatch",
            "The pending interrupt no longer matches the stored agent run.",
        )


class AgentPatternRepository:
    """Persistence for agent workflows and runs."""

    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def _get_session(self) -> AsyncSession:
        if self._session is not None:
            return self._session
        return async_session_maker()

    async def seed_builtin_workflows(self) -> None:
        validator = TemplateValidator()
        session = await self._get_session()
        async with session.begin():
            for workflow_def in load_builtin_workflows():
                spec_json = workflow_def["spec_json"]
                validation_result = validator.validate(spec_json)
                builtin_key = workflow_def["builtin_key"]
                metadata = {
                    "source": "builtin",
                    "builtin_key": builtin_key,
                }

                workflow = await self._get_workflow_by_builtin_key(session, builtin_key)
                if workflow is None:
                    workflow = await self._get_workflow_by_name(session, workflow_def["name"])
                if workflow is None:
                    workflow = AgentWorkflow(
                        name=workflow_def["name"],
                        description=workflow_def["description"],
                        visibility=workflow_def["visibility"],
                        is_builtin=workflow_def["is_builtin"],
                        schema_version=spec_json["schema_version"],
                        spec_json=spec_json,
                        validation_result_json=validation_result,
                        metadata_json=metadata,
                    )
                    session.add(workflow)
                else:
                    if not workflow.is_builtin and workflow.visibility != "deleted":
                        raise ValueError(f"agent workflow name already exists: {workflow_def['name']}")
                    workflow.name = workflow_def["name"]
                    workflow.description = workflow_def["description"]
                    workflow.visibility = workflow_def["visibility"]
                    workflow.is_builtin = workflow_def["is_builtin"]
                    workflow.schema_version = spec_json["schema_version"]
                    replace_jsonb_field(workflow, "spec_json", spec_json)
                    replace_jsonb_field(workflow, "validation_result_json", validation_result)
                    replace_jsonb_field(workflow, "metadata_json", metadata)
                    workflow.updated_at = utc_now()

    async def _get_workflow_by_name(self, session: AsyncSession, name: str) -> Optional[AgentWorkflow]:
        result = await session.execute(select(AgentWorkflow).where(AgentWorkflow.name == name))
        return result.scalars().first()

    async def _get_workflow_by_builtin_key(self, session: AsyncSession, builtin_key: str) -> Optional[AgentWorkflow]:
        result = await session.execute(
            select(AgentWorkflow).where(AgentWorkflow.metadata_json["builtin_key"].astext == builtin_key)
        )
        workflow = result.scalars().first()
        if workflow is not None:
            return workflow
        result = await session.execute(
            select(AgentWorkflow).where(AgentWorkflow.spec_json["pattern_type"].astext == builtin_key)
        )
        return result.scalars().first()

    async def list_workflows(self, *, include_custom: bool = False) -> list[AgentWorkflow]:
        session = await self._get_session()
        async with session.begin():
            visibility_filter = (
                AgentWorkflow.is_builtin.is_(True)
                if not include_custom
                else or_(
                    AgentWorkflow.is_builtin.is_(True),
                    AgentWorkflow.visibility.in_(["public", "internal"]),
                )
            )
            result = await session.execute(
                select(AgentWorkflow)
                .where(visibility_filter)
                .order_by(AgentWorkflow.name.asc())
            )
            return list(result.scalars().all())

    async def mark_custom_workflow_deleted(self, workflow_id: str) -> Optional[AgentWorkflow]:
        session = await self._get_session()
        async with session.begin():
            workflow = await session.get(AgentWorkflow, workflow_id)
            if workflow is None or workflow.is_builtin:
                if workflow is not None and workflow.is_builtin:
                    raise ValueError("built-in agent workflows cannot be deleted")
                return None
            workflow.visibility = "deleted"
            workflow.updated_at = utc_now()
            await session.flush()
            return workflow

    async def get_workflow(self, workflow_id: str, *, include_custom: bool = False) -> Optional[AgentWorkflow]:
        session = await self._get_session()
        async with session.begin():
            workflow = await session.get(AgentWorkflow, workflow_id)
            if workflow is None and workflow_id in builtin_workflow_keys():
                workflow = await self._get_workflow_by_builtin_key(session, workflow_id)
            if not workflow:
                return None
            if not include_custom and not workflow.is_builtin:
                return None
            if include_custom and not workflow.is_builtin and workflow.visibility not in {"public", "internal"}:
                return None
            return workflow

    async def save_custom_workflow(
        self,
        *,
        workflow_id: Optional[str],
        name: str,
        spec_json: Dict[str, Any],
        description: str = "",
        visibility: str = "internal",
    ) -> AgentWorkflow:
        """Create or update a mutable internal/custom workflow spec."""
        if workflow_id in builtin_workflow_keys():
            raise ValueError("built-in agent workflows cannot be authored through the internal path")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(spec_json, dict):
            raise TemplateValidationError("spec must be an object")
        if spec_json.get("schema_version") != 2:
            raise TemplateValidationError("internal custom agent workflow specs must use schema_version 2")

        validation_result = TemplateValidator().validate(spec_json)
        session = await self._get_session()
        async with session.begin():
            workflow = await session.get(AgentWorkflow, workflow_id) if workflow_id else None
            existing_named_workflow = await self._get_workflow_by_name(session, name)
            if existing_named_workflow is not None and (workflow is None or existing_named_workflow.id != workflow.id):
                raise ValueError(f"agent workflow name already exists: {name}")
            previous_metadata = workflow.metadata_json if workflow and isinstance(workflow.metadata_json, dict) else {}
            metadata = {
                **previous_metadata,
                "source": "custom",
            }
            if workflow is None:
                workflow = AgentWorkflow(
                    name=name,
                    description=description,
                    visibility=visibility,
                    is_builtin=False,
                    schema_version=2,
                    spec_json=spec_json,
                    validation_result_json=validation_result,
                    metadata_json=metadata,
                )
                session.add(workflow)
            else:
                if workflow.is_builtin:
                    raise ValueError("built-in agent workflows cannot be authored through the internal path")
                workflow.name = name
                workflow.description = description
                workflow.visibility = visibility
                workflow.is_builtin = False
                workflow.schema_version = 2
                replace_jsonb_field(workflow, "spec_json", spec_json)
                replace_jsonb_field(workflow, "validation_result_json", validation_result)
                replace_jsonb_field(workflow, "metadata_json", metadata)
                workflow.updated_at = utc_now()

            await session.flush()
            return workflow

    async def get_run(self, run_id: str) -> Optional[AgentRun]:
        session = await self._get_session()
        async with session.begin():
            return await session.get(AgentRun, run_id)

    async def list_runs_for_thread(
        self,
        thread_id: str,
        *,
        limit: int = 20,
        status: Optional[str] = None,
    ) -> list[AgentRun]:
        session = await self._get_session()
        bounded_limit = max(1, min(int(limit), 100))
        async with session.begin():
            query = select(AgentRun).where(AgentRun.thread_id == thread_id)
            if status:
                query = query.where(AgentRun.status == status)
            result = await session.execute(
                query.order_by(AgentRun.started_at.desc(), AgentRun.id.desc()).limit(bounded_limit)
            )
            return list(result.scalars().all())

    async def prune_runs_before(
        self,
        cutoff: datetime,
        *,
        statuses: list[str],
        thread_id: Optional[str] = None,
        limit: int = 1000,
    ) -> list[str]:
        """Delete old run records matching explicit terminal statuses."""

        if not statuses:
            raise ValueError("statuses must contain at least one status")
        bounded_limit = max(1, min(int(limit), 1000))
        session = await self._get_session()
        async with session.begin():
            query = (
                select(AgentRun.id)
                .where(AgentRun.started_at < cutoff)
                .where(AgentRun.status.in_(statuses))
            )
            if thread_id is not None:
                query = query.where(AgentRun.thread_id == thread_id)
            query = query.order_by(AgentRun.started_at.asc(), AgentRun.id.asc()).limit(bounded_limit)
            result = await session.execute(query)
            run_ids = list(result.scalars().all())
            if not run_ids:
                return []
            await session.execute(delete(AgentRun).where(AgentRun.id.in_(run_ids)))
            return run_ids

    async def prune_checkpoints_for_runs_before(
        self,
        cutoff: datetime,
        *,
        statuses: Optional[list[str]] = None,
        thread_id: Optional[str] = None,
        limit: int = 1000,
        checkpointer: Any = None,
    ) -> list[str]:
        """Delete LangGraph checkpoints for old terminal runs only."""

        requested_statuses = statuses or sorted(CHECKPOINT_PRUNABLE_RUN_STATUSES)
        if not requested_statuses:
            raise ValueError("statuses must contain at least one status")
        invalid_statuses = sorted(set(requested_statuses) - CHECKPOINT_PRUNABLE_RUN_STATUSES)
        if invalid_statuses:
            raise ValueError(
                "checkpoint cleanup is only allowed for terminal run statuses; "
                f"invalid statuses: {', '.join(invalid_statuses)}"
            )
        bounded_limit = max(1, min(int(limit), 1000))
        session = await self._get_session()
        async with session.begin():
            query = (
                select(AgentRun.checkpoint_thread_id)
                .where(AgentRun.started_at < cutoff)
                .where(AgentRun.status.in_(requested_statuses))
                .where(AgentRun.checkpoint_thread_id.isnot(None))
            )
            if thread_id is not None:
                query = query.where(AgentRun.thread_id == thread_id)
            result = await session.execute(
                query.order_by(AgentRun.started_at.asc(), AgentRun.id.asc()).limit(bounded_limit)
            )
            checkpoint_thread_ids = list(result.scalars().all())
        return await delete_agent_checkpoints(checkpoint_thread_ids, checkpointer=checkpointer)

    async def fail_stale_running_runs(
        self,
        cutoff: datetime,
        *,
        thread_id: Optional[str] = None,
        limit: int = 1000,
    ) -> list[str]:
        """Mark old running runs failed after a process crash or restart."""

        bounded_limit = max(1, min(int(limit), 1000))
        session = await self._get_session()
        async with session.begin():
            query = (
                select(AgentRun)
                .where(AgentRun.started_at < cutoff)
                .where(AgentRun.status == RUN_STATUS_RUNNING)
            )
            if thread_id is not None:
                query = query.where(AgentRun.thread_id == thread_id)
            query = query.order_by(AgentRun.started_at.asc(), AgentRun.id.asc()).limit(bounded_limit)
            result = await session.execute(query)
            runs = list(result.scalars().all())
            completed_at = utc_now()
            for run in runs:
                run.status = "failed"
                run.completed_at = completed_at
                replace_jsonb_field(
                    run,
                    "error_json",
                    {
                        "code": "agent_run_stale",
                        "raw_message": "Agent run was still running past the stale-run cutoff.",
                        "retryable": True,
                    },
                )
                metrics = dict(run.metrics_json or {})
                metrics["error_count"] = max(int(metrics.get("error_count") or 0), 1)
                replace_jsonb_field(run, "metrics_json", metrics)
            return [run.id for run in runs]

    async def list_chat_turns_for_run(self, run_id: str) -> list[ChatTurn]:
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(
                select(ChatTurn)
                .where(ChatTurn.agent_run_id == run_id)
                .order_by(ChatTurn.agent_run_sequence.asc(), ChatTurn.created_at.asc(), ChatTurn.id.asc())
            )
            return list(result.scalars().all())

    async def create_run(
        self,
        *,
        thread_id: str,
        workflow_id: str,
        resolved_spec_json: Dict[str, Any],
        user_id: Optional[str] = None,
        checkpoint_thread_id: Optional[str] = None,
    ) -> AgentRun:
        run_id = str(uuid.uuid4())
        run = AgentRun(
            id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            workflow_id=workflow_id,
            run_metadata_json={},
            resolved_spec_json=resolved_spec_json,
            status=RUN_STATUS_RUNNING,
            checkpoint_thread_id=checkpoint_thread_id or run_id,
            started_at=utc_now(),
        )
        session = await self._get_session()
        async with session.begin():
            session.add(run)
            await session.flush()
            await session.refresh(run)
        return run

    async def mark_run_awaiting_human(
        self,
        run_id: str,
        pending_interrupt_json: Dict[str, Any],
        *,
        metrics_json: Optional[Dict[str, Any]] = None,
        debug_trace_json: Optional[Dict[str, Any]] = None,
        requested_at: Optional[datetime] = None,
    ) -> Optional[AgentRun]:
        """Pause a run on a bounded pending-interrupt payload."""

        session = await self._get_session()
        async with session.begin():
            run = await session.get(AgentRun, run_id)
            if not run:
                return None
            pending_interrupt = normalize_pending_interrupt_payload(
                {
                    **pending_interrupt_json,
                    "compatibility": _run_interrupt_compatibility(run),
                },
                requested_at=requested_at,
            )
            run.status = RUN_STATUS_AWAITING_HUMAN
            run.completed_at = None
            replace_jsonb_field(run, "pending_interrupt_json", pending_interrupt)
            if metrics_json is not None:
                replace_jsonb_field(run, "metrics_json", metrics_json)
            trace_payload = debug_trace_json if debug_trace_json is not None else run.debug_trace_json
            if isinstance(trace_payload, dict):
                replace_jsonb_field(
                    run,
                    "debug_trace_json",
                    append_interrupt_event_to_debug_payload(
                        trace_payload,
                        pending_interrupt,
                        event_name="interrupt.requested",
                        run_status=RUN_STATUS_AWAITING_HUMAN,
                    ),
                )
            await session.flush()
            await session.refresh(run)
            return run

    def _build_interrupt_decision(
        self,
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
            decision["edited_payload"] = _compact_interrupt_value(edited_payload)
        if selected_option_ids is not None:
            decision["selected_option_ids"] = _compact_interrupt_value(selected_option_ids)
        if client_metadata is not None:
            decision["client_metadata"] = _compact_interrupt_value(client_metadata)
        return decision

    def _validate_pending_interrupt_request(
        self,
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

        if action == "approve_selected":
            valid_option_ids = _option_ids(interrupt)
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
            selection_mode = str(interrupt.get("selection_mode") or "single")
            if selection_mode == "single" and len(selected_option_ids) != 1:
                raise AgentRunInterruptError(
                    "interrupt_selection_count_invalid",
                    "This interrupt requires exactly one selected option.",
                    http_status=400,
                )

    async def resolve_pending_interrupt(
        self,
        run_id: str,
        *,
        interrupt_id: str,
        action: str,
        edited_payload: Optional[Dict[str, Any]] = None,
        client_metadata: Optional[Dict[str, Any]] = None,
        selected_option_ids: Optional[list[str]] = None,
        resume_token: Optional[str] = None,
        resume_version: Optional[int] = None,
        expected_thread_id: Optional[str] = None,
        decided_at: Optional[datetime] = None,
    ) -> Optional[InterruptResolutionResult]:
        """Resolve one pending interrupt atomically and idempotently."""

        if action not in RESUME_ACTIONS and action != "reject":
            raise AgentRunInterruptError(
                "invalid_interrupt_action",
                f"Unsupported interrupt action: {action}",
                http_status=400,
            )

        now = decided_at or utc_now()
        session = await self._get_session()
        async with session.begin():
            result = await session.execute(select(AgentRun).where(AgentRun.id == run_id).with_for_update())
            run = result.scalar_one_or_none()
            if not run:
                return None
            if expected_thread_id is not None and run.thread_id != expected_thread_id:
                raise AgentRunInterruptError(
                    "run_thread_mismatch",
                    "The agent run does not belong to the requested thread.",
                    http_status=404,
                )

            interrupt = _pending_interrupt_from_run(run)
            if not interrupt:
                raise AgentRunInterruptError(
                    "no_pending_interrupt",
                    "This agent run does not have an interrupt to resolve.",
                    http_status=404,
                )
            current_status = str(interrupt.get("status") or INTERRUPT_STATUS_PENDING)
            if current_status in TERMINAL_INTERRUPT_STATUSES:
                if _terminal_decision_matches(interrupt, action=action, interrupt_id=interrupt_id):
                    return InterruptResolutionResult(
                        run=run,
                        outcome=current_status,
                        interrupt=interrupt,
                        duplicate=True,
                    )
                raise AgentRunInterruptError(
                    "interrupt_already_resolved",
                    "This interrupt has already been resolved.",
                )
            if run.status != RUN_STATUS_AWAITING_HUMAN or current_status != INTERRUPT_STATUS_PENDING:
                raise AgentRunInterruptError(
                    "interrupt_not_pending",
                    "This agent run is not awaiting a human decision.",
                )

            _validate_interrupt_compatibility(interrupt, run)
            self._validate_pending_interrupt_request(
                interrupt,
                interrupt_id=interrupt_id,
                action=action,
                resume_token=resume_token,
                resume_version=resume_version,
                selected_option_ids=selected_option_ids,
            )

            decision = self._build_interrupt_decision(
                interrupt_id=interrupt_id,
                action=action,
                decided_at=now,
                edited_payload=edited_payload,
                client_metadata=client_metadata,
                resume_version=resume_version,
                selected_option_ids=selected_option_ids,
            )
            if isinstance(run.debug_trace_json, dict):
                replace_jsonb_field(
                    run,
                    "debug_trace_json",
                    append_runtime_event_to_debug_payload(
                        run.debug_trace_json,
                        "resume.requested",
                        attributes={
                            "askpdf.interrupt.id": interrupt_id,
                            "askpdf.resume.action": action,
                            "askpdf.interrupt.resume_version": resume_version or interrupt.get("resume_version"),
                            "askpdf.checkpoint.thread_id": run.checkpoint_thread_id,
                        },
                    ),
                )

            if _interrupt_expired(interrupt, now):
                interrupt["status"] = INTERRUPT_STATUS_EXPIRED
                interrupt["decision"] = {
                    **decision,
                    "action": "expire",
                    "requested_action": action,
                }
                run.status = RUN_STATUS_EXPIRED
                run.completed_at = now
                replace_jsonb_field(
                    run,
                    "error_json",
                    {
                        "code": "agent_run_interrupt_expired",
                        "raw_message": "Agent run interrupt expired before a human decision was accepted.",
                        "retryable": False,
                    },
                )
                outcome = INTERRUPT_STATUS_EXPIRED
            elif action == "reject" and interrupt.get("reject_behavior") == "resume":
                interrupt["status"] = INTERRUPT_STATUS_RESUMED
                interrupt["decision"] = decision
                run.status = RUN_STATUS_RUNNING
                run.completed_at = None
                outcome = INTERRUPT_STATUS_RESUMED
            elif action == "reject":
                interrupt["status"] = INTERRUPT_STATUS_REJECTED
                interrupt["decision"] = decision
                run.status = RUN_STATUS_REJECTED
                run.completed_at = now
                replace_jsonb_field(
                    run,
                    "error_json",
                    {
                        "code": "agent_run_rejected_by_human",
                        "raw_message": "Agent run was rejected by a human reviewer.",
                        "retryable": False,
                    },
                )
                outcome = INTERRUPT_STATUS_REJECTED
            else:
                interrupt["status"] = INTERRUPT_STATUS_RESUMED
                interrupt["decision"] = decision
                run.status = RUN_STATUS_RUNNING
                run.completed_at = None
                outcome = INTERRUPT_STATUS_RESUMED

            metrics = dict(run.metrics_json or {})
            metrics["interrupt_resolution_count"] = int(metrics.get("interrupt_resolution_count") or 0) + 1
            metrics["last_interrupt_action"] = interrupt["decision"].get("action")
            replace_jsonb_field(run, "metrics_json", metrics)
            replace_jsonb_field(run, "pending_interrupt_json", interrupt)
            if isinstance(run.debug_trace_json, dict):
                debug_payload = append_interrupt_event_to_debug_payload(
                    run.debug_trace_json,
                    interrupt,
                    run_status=run.status,
                    completed_at=run.completed_at,
                )
                if outcome == INTERRUPT_STATUS_RESUMED:
                    debug_payload = append_runtime_event_to_debug_payload(
                        debug_payload,
                        "resume.applied",
                        attributes={
                            "askpdf.interrupt.id": interrupt_id,
                            "askpdf.resume.action": action,
                            "askpdf.interrupt.resume_version": resume_version or interrupt.get("resume_version"),
                            "askpdf.checkpoint.thread_id": run.checkpoint_thread_id,
                        },
                        run_status=run.status,
                        completed_at=run.completed_at,
                    )
                replace_jsonb_field(run, "debug_trace_json", debug_payload)
            await session.flush()
            await session.refresh(run)
            return InterruptResolutionResult(run=run, outcome=outcome, interrupt=interrupt)

    async def expire_pending_interrupts(
        self,
        *,
        now: Optional[datetime] = None,
        thread_id: Optional[str] = None,
        limit: int = 1000,
    ) -> list[str]:
        """Mark expired awaiting-human runs independently from stale running runs."""

        cutoff = now or utc_now()
        bounded_limit = max(1, min(int(limit), 1000))
        session = await self._get_session()
        async with session.begin():
            query = select(AgentRun).where(AgentRun.status == RUN_STATUS_AWAITING_HUMAN)
            if thread_id is not None:
                query = query.where(AgentRun.thread_id == thread_id)
            result = await session.execute(
                query.order_by(AgentRun.started_at.asc(), AgentRun.id.asc()).limit(bounded_limit).with_for_update()
            )
            runs = list(result.scalars().all())
            expired_ids: list[str] = []
            for run in runs:
                interrupt = _pending_interrupt_from_run(run)
                if not interrupt or str(interrupt.get("status") or INTERRUPT_STATUS_PENDING) != INTERRUPT_STATUS_PENDING:
                    continue
                if not _interrupt_expired(interrupt, cutoff):
                    continue
                interrupt["status"] = INTERRUPT_STATUS_EXPIRED
                interrupt["decision"] = {
                    "interrupt_id": interrupt.get("interrupt_id"),
                    "action": "expire",
                    "decided_at": iso_utc_z(cutoff),
                }
                run.status = RUN_STATUS_EXPIRED
                run.completed_at = cutoff
                replace_jsonb_field(run, "pending_interrupt_json", interrupt)
                replace_jsonb_field(
                    run,
                    "error_json",
                    {
                        "code": "agent_run_interrupt_expired",
                        "raw_message": "Agent run interrupt expired before a human decision was accepted.",
                        "retryable": False,
                    },
                )
                metrics = dict(run.metrics_json or {})
                metrics["interrupt_expired_count"] = int(metrics.get("interrupt_expired_count") or 0) + 1
                replace_jsonb_field(run, "metrics_json", metrics)
                if isinstance(run.debug_trace_json, dict):
                    replace_jsonb_field(
                        run,
                        "debug_trace_json",
                        append_interrupt_event_to_debug_payload(
                            run.debug_trace_json,
                            interrupt,
                            run_status=run.status,
                            completed_at=run.completed_at,
                        ),
                    )
                expired_ids.append(run.id)
            return expired_ids

    async def complete_run(
        self,
        run_id: str,
        *,
        status: str,
        metrics_json: Optional[Dict[str, Any]] = None,
        error_json: Optional[Dict[str, Any]] = None,
        debug_trace_json: Optional[Dict[str, Any]] = None,
        completed_at: Optional[datetime] = None,
    ) -> Optional[AgentRun]:
        session = await self._get_session()
        async with session.begin():
            run = await session.get(AgentRun, run_id)
            if not run:
                return None
            run.status = status
            run.completed_at = completed_at or utc_now()
            replace_jsonb_field(run, "metrics_json", metrics_json or {})
            if error_json is not None:
                replace_jsonb_field(run, "error_json", error_json)
            if debug_trace_json is not None:
                replace_jsonb_field(run, "debug_trace_json", debug_trace_json)
            await session.flush()
            await session.refresh(run)
            return run

    async def set_run_debug_trace(
        self,
        run_id: str,
        debug_trace_json: Dict[str, Any],
    ) -> Optional[AgentRun]:
        session = await self._get_session()
        async with session.begin():
            run = await session.get(AgentRun, run_id)
            if not run:
                return None
            replace_jsonb_field(run, "debug_trace_json", debug_trace_json)
            await session.flush()
            await session.refresh(run)
            return run
