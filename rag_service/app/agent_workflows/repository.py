from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.agent_workflows.debug_trace import append_interrupt_event_to_debug_payload, append_runtime_event_to_debug_payload
from app.agent_workflows.interrupts import (
    INTERRUPT_STATUS_EXPIRED,
    INTERRUPT_STATUS_PENDING,
    INTERRUPT_STATUS_REJECTED,
    INTERRUPT_STATUS_RESUMED,
    RESUME_ACTIONS,
    TERMINAL_INTERRUPT_STATUSES,
    AgentRunInterruptError,
    InterruptResolutionResult,
    build_interrupt_decision,
    interrupt_expired,
    normalize_pending_interrupt_payload,
    pending_interrupt_from_run,
    run_interrupt_resume_guard,
    terminal_decision_matches,
    validate_interrupt_resume_guard,
    validate_pending_interrupt_request,
)
from app.agent_workflows.builtin_workflows import builtin_workflow_keys, load_builtin_workflows
from app.agent_workflows.run_cleanup import (
    fail_stale_running_runs as cleanup_fail_stale_running_runs,
    prune_checkpoints_for_runs_before as cleanup_prune_checkpoints_for_runs_before,
    prune_runs_before as cleanup_prune_runs_before,
)
from app.agent_workflows.validator import WorkflowValidationError, WorkflowValidator
from app.db.connection_sqlmodel import async_session_maker
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import (
    AgentWorkflow,
    AgentRun,
    ChatTurn,
)
from app.time_utils import iso_utc_z, utc_now


RUN_STATUS_RUNNING = "running"
RUN_STATUS_AWAITING_HUMAN = "awaiting_human"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_CLARIFICATION = "clarification"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_REJECTED = "rejected"
RUN_STATUS_EXPIRED = "expired"


@dataclass
class AgentWorkflowVersion:
    id: str
    workflow_id: str
    version: int
    schema_version: int
    spec_json: Dict[str, Any]
    validation_result_json: Dict[str, Any]
    metadata_json: Dict[str, Any]


def _workflow_version(workflow: AgentWorkflow) -> AgentWorkflowVersion:
    metadata = workflow.metadata_json if isinstance(workflow.metadata_json, dict) else {}
    version = workflow.version
    version_id = str(metadata.get("version_id") or f"{workflow.id}:v{version}")
    return AgentWorkflowVersion(
        id=version_id,
        workflow_id=workflow.id,
        version=version,
        schema_version=workflow.schema_version,
        spec_json=workflow.spec_json,
        validation_result_json=workflow.validation_result_json,
        metadata_json=metadata,
    )




class AgentWorkflowRepository:
    """Persistence for agent workflows and runs."""

    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session

    async def _get_session(self) -> AsyncSession:
        if self._session is not None:
            return self._session
        return async_session_maker()

    async def seed_builtin_workflows(self) -> None:
        validator = WorkflowValidator()
        session = await self._get_session()
        async with session.begin():
            for workflow_def in load_builtin_workflows():
                spec_json = workflow_def["spec_json"]
                validation_result = validator.validate(spec_json)
                builtin_key = workflow_def["builtin_key"]
                metadata = {
                    "source": "builtin",
                    "builtin_key": builtin_key,
                    "version": spec_json.get("version") or 2,
                    "version_id": f"{builtin_key}:v{spec_json.get('version') or 2}",
                }

                workflow = await self._get_workflow_by_builtin_key(session, builtin_key)
                if workflow is None:
                    workflow = await self._get_workflow_by_name(session, workflow_def["name"])
                if workflow is None:
                    workflow = AgentWorkflow(
                        id=builtin_key,
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
            select(AgentWorkflow).where(AgentWorkflow.spec_json["workflow_id"].astext == builtin_key)
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

    async def get_workflow_with_current_version(
        self,
        workflow_id: str,
        *,
        include_custom: bool = False,
    ) -> tuple[Optional[AgentWorkflow], Optional[AgentWorkflowVersion]]:
        workflow = await self.get_workflow(workflow_id, include_custom=include_custom)
        if workflow is None:
            return None, None
        return workflow, _workflow_version(workflow)

    async def get_workflow_version(
        self,
        workflow_id: str,
        version: int,
        *,
        include_custom: bool = False,
    ) -> tuple[Optional[AgentWorkflow], Optional[AgentWorkflowVersion]]:
        workflow, current_version = await self.get_workflow_with_current_version(
            workflow_id,
            include_custom=include_custom,
        )
        if current_version is None or current_version.version != int(version):
            return None, None
        return workflow, current_version

    async def save_custom_workflow(
        self,
        *,
        workflow_id: Optional[str],
        name: str,
        spec_json: Dict[str, Any],
        description: str = "",
        visibility: str = "internal",
        increment_version: bool = True,
    ) -> AgentWorkflow:
        """Create or update a mutable internal/custom workflow spec."""
        if workflow_id in builtin_workflow_keys():
            raise ValueError("built-in agent workflows cannot be authored through the internal path")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(spec_json, dict):
            raise WorkflowValidationError("spec must be an object")
        if spec_json.get("schema_version") != 2:
            raise WorkflowValidationError("internal custom agent workflow specs must use schema_version 2")

        validation_result = WorkflowValidator().validate(spec_json)
        session = await self._get_session()
        async with session.begin():
            workflow = await session.get(AgentWorkflow, workflow_id) if workflow_id else None
            existing_named_workflow = await self._get_workflow_by_name(session, name)
            if existing_named_workflow is not None and (workflow is None or existing_named_workflow.id != workflow.id):
                raise ValueError(f"agent workflow name already exists: {name}")
            previous_metadata = workflow.metadata_json if workflow and isinstance(workflow.metadata_json, dict) else {}
            previous_version = previous_metadata.get("version")
            try:
                next_version = int(previous_version) + 1 if workflow is not None and increment_version else int(previous_version or 1)
            except (TypeError, ValueError):
                next_version = 1
            workflow_key = workflow_id or spec_json.get("workflow_id") or name
            metadata = {
                **previous_metadata,
                "source": "custom",
                "version": next_version,
                "version_id": f"{workflow_key}:v{next_version}",
            }
            if workflow is None:
                workflow = AgentWorkflow(
                    id=workflow_id or str(uuid.uuid4()),
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

    async def save_internal_workflow_version(
        self,
        *,
        workflow_id: str,
        name: str,
        spec_json: Dict[str, Any],
        description: str = "",
        visibility: str = "internal",
        changelog: str = "",
        increment_version: bool = True,
    ) -> tuple[AgentWorkflow, AgentWorkflowVersion]:
        workflow = await self.save_custom_workflow(
            workflow_id=workflow_id,
            name=name,
            description=description,
            visibility=visibility,
            spec_json=spec_json,
            increment_version=increment_version,
        )
        return workflow, _workflow_version(workflow)

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

        session = await self._get_session()
        return await cleanup_prune_runs_before(
            session,
            cutoff,
            statuses=statuses,
            thread_id=thread_id,
            limit=limit,
        )

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

        session = await self._get_session()
        return await cleanup_prune_checkpoints_for_runs_before(
            session,
            cutoff,
            statuses=statuses,
            thread_id=thread_id,
            limit=limit,
            checkpointer=checkpointer,
        )

    async def fail_stale_running_runs(
        self,
        cutoff: datetime,
        *,
        thread_id: Optional[str] = None,
        limit: int = 1000,
    ) -> list[str]:
        """Mark old running runs failed after a process crash or restart."""

        session = await self._get_session()
        return await cleanup_fail_stale_running_runs(
            session,
            cutoff,
            thread_id=thread_id,
            limit=limit,
        )

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
        workflow_version_id: Optional[str] = None,
        workflow_version: Optional[int] = None,
        resolved_spec_json: Dict[str, Any],
        user_id: Optional[str] = None,
        checkpoint_thread_id: Optional[str] = None,
    ) -> AgentRun:
        run_metadata: Dict[str, Any] = {}
        if workflow_version_id is not None:
            run_metadata["workflow_version_id"] = workflow_version_id
        if workflow_version is not None:
            run_metadata["workflow_version"] = workflow_version
        run_id = str(uuid.uuid4())
        run = AgentRun(
            id=run_id,
            thread_id=thread_id,
            user_id=user_id,
            workflow_id=workflow_id,
            run_metadata_json=run_metadata,
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
                    "resume_guard": run_interrupt_resume_guard(run),
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

            interrupt = pending_interrupt_from_run(run)
            if not interrupt:
                raise AgentRunInterruptError(
                    "no_pending_interrupt",
                    "This agent run does not have an interrupt to resolve.",
                    http_status=404,
                )
            current_status = str(interrupt.get("status") or INTERRUPT_STATUS_PENDING)
            if current_status in TERMINAL_INTERRUPT_STATUSES:
                if terminal_decision_matches(interrupt, action=action, interrupt_id=interrupt_id):
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

            validate_interrupt_resume_guard(interrupt, run)
            validate_pending_interrupt_request(
                interrupt,
                interrupt_id=interrupt_id,
                action=action,
                resume_token=resume_token,
                resume_version=resume_version,
                selected_option_ids=selected_option_ids,
            )

            decision = build_interrupt_decision(
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

            if interrupt_expired(interrupt, now):
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
                        "code": "agent_runinterrupt_expired",
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
                interrupt = pending_interrupt_from_run(run)
                if not interrupt or str(interrupt.get("status") or INTERRUPT_STATUS_PENDING) != INTERRUPT_STATUS_PENDING:
                    continue
                if not interrupt_expired(interrupt, cutoff):
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
                        "code": "agent_runinterrupt_expired",
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
