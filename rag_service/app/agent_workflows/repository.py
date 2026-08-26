from __future__ import annotations

import functools
import inspect
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.agent_workflows.debug_trace import append_interrupt_event_to_debug_payload, append_runtime_event_to_debug_payload
from app.agent_workflows.enums import AgentRunResumeAction, HitlRejectBehavior
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
from app.agent_workflows.run_cleanup import (
    fail_stale_running_runs as cleanup_fail_stale_running_runs,
    prune_checkpoints_for_runs_before as cleanup_prune_checkpoints_for_runs_before,
    prune_runs_before as cleanup_prune_runs_before,
)
from app.agent_workflows.run_store import (
    append_run_event as run_store_append_run_event,
    complete_run as run_store_complete_run,
    create_run as run_store_create_run,
    delete_run as run_store_delete_run,
    get_run as run_store_get_run,
    list_chat_turns_for_run as run_store_list_chat_turns_for_run,
    list_runs_for_thread as run_store_list_runs_for_thread,
    list_run_events as run_store_list_run_events,
    set_run_debug_trace as run_store_set_run_debug_trace,
)
from app.agent_workflows.workflow_store import (
    AgentWorkflowVersion,
    get_workflow as workflow_store_get_workflow,
    get_workflow_by_builtin_key as workflow_store_get_workflow_by_builtin_key,
    get_workflow_by_name as workflow_store_get_workflow_by_name,
    get_workflow_version as workflow_store_get_workflow_version,
    get_workflow_with_current_version as workflow_store_get_workflow_with_current_version,
    list_workflows as workflow_store_list_workflows,
    mark_custom_workflow_deleted as workflow_store_mark_custom_workflow_deleted,
    save_custom_workflow as workflow_store_save_custom_workflow,
    save_internal_workflow_version as workflow_store_save_internal_workflow_version,
    seed_builtin_workflows as workflow_store_seed_builtin_workflows,
)
from app.db.connection_sqlmodel import async_session_maker
from app.db.jsonb_utils import replace_jsonb_field
from app.db.models_sqlmodel import (
    AgentRunStatus,
    AgentTask,
    AgentWorkflow,
    AgentRun,
    ChatTurn,
)
from app.time_utils import iso_utc_z, utc_now


RUN_STATUS_RUNNING = AgentRunStatus.RUNNING.value
RUN_STATUS_AWAITING_HUMAN = AgentRunStatus.AWAITING_HUMAN.value
RUN_STATUS_COMPLETED = AgentRunStatus.COMPLETED.value
RUN_STATUS_CLARIFICATION = AgentRunStatus.CLARIFICATION.value
RUN_STATUS_FAILED = AgentRunStatus.FAILED.value
RUN_STATUS_REJECTED = AgentRunStatus.REJECTED.value
RUN_STATUS_EXPIRED = AgentRunStatus.EXPIRED.value


class AgentWorkflowRepository:
    """Persistence for agent workflows and runs."""

    def __init__(self, session: Optional[AsyncSession] = None):
        self._session = session
        self._operation_session: ContextVar[AsyncSession | None] = ContextVar(
            f"agent_workflow_repository_session_{id(self)}",
            default=None,
        )

    async def _get_session(self) -> AsyncSession:
        if self._session is not None:
            return self._session
        session = self._operation_session.get()
        if session is None:
            raise RuntimeError("AgentWorkflowRepository operation has no managed database session")
        return session

    async def seed_builtin_workflows(self) -> None:
        session = await self._get_session()
        await workflow_store_seed_builtin_workflows(session)

    async def _get_workflow_by_name(self, session: AsyncSession, name: str) -> Optional[AgentWorkflow]:
        return await workflow_store_get_workflow_by_name(session, name)

    async def _get_workflow_by_builtin_key(
        self,
        session: AsyncSession,
        builtin_key: str,
    ) -> Optional[AgentWorkflow]:
        return await workflow_store_get_workflow_by_builtin_key(session, builtin_key)

    async def list_workflows(self, *, include_custom: bool = False) -> list[AgentWorkflow]:
        session = await self._get_session()
        return await workflow_store_list_workflows(session, include_custom=include_custom)

    async def mark_custom_workflow_deleted(self, workflow_id: str) -> Optional[AgentWorkflow]:
        session = await self._get_session()
        return await workflow_store_mark_custom_workflow_deleted(session, workflow_id)

    async def get_workflow(self, workflow_id: str, *, include_custom: bool = False) -> Optional[AgentWorkflow]:
        session = await self._get_session()
        return await workflow_store_get_workflow(session, workflow_id, include_custom=include_custom)

    async def get_workflow_with_current_version(
        self,
        workflow_id: str,
        *,
        include_custom: bool = False,
    ) -> tuple[Optional[AgentWorkflow], Optional[AgentWorkflowVersion]]:
        session = await self._get_session()
        return await workflow_store_get_workflow_with_current_version(
            session,
            workflow_id,
            include_custom=include_custom,
        )

    async def get_workflow_version(
        self,
        workflow_id: str,
        version: int,
        *,
        include_custom: bool = False,
    ) -> tuple[Optional[AgentWorkflow], Optional[AgentWorkflowVersion]]:
        session = await self._get_session()
        return await workflow_store_get_workflow_version(
            session,
            workflow_id,
            version,
            include_custom=include_custom,
        )

    async def save_custom_workflow(
        self,
        *,
        workflow_id: Optional[str],
        name: str,
        spec_json: Dict[str, Any],
        framework: Optional[str] = None,
        builder_id: Optional[str] = None,
        description: str = "",
        visibility: str = "internal",
    ) -> AgentWorkflow:
        session = await self._get_session()
        return await workflow_store_save_custom_workflow(
            session,
            workflow_id=workflow_id,
            name=name,
            spec_json=spec_json,
            framework=framework,
            builder_id=builder_id,
            description=description,
            visibility=visibility,
        )

    async def save_internal_workflow_version(
        self,
        *,
        workflow_id: str,
        name: str,
        spec_json: Dict[str, Any],
        framework: str,
        builder_id: str,
        description: str = "",
        visibility: str = "internal",
        changelog: str = "",
    ) -> tuple[AgentWorkflow, AgentWorkflowVersion]:
        session = await self._get_session()
        return await workflow_store_save_internal_workflow_version(
            session,
            workflow_id=workflow_id,
            name=name,
            spec_json=spec_json,
            framework=framework,
            builder_id=builder_id,
            description=description,
            visibility=visibility,
            changelog=changelog,
        )

    async def get_run(self, run_id: str) -> Optional[AgentRun]:
        session = await self._get_session()
        return await run_store_get_run(session, run_id)

    async def list_runs_for_thread(
        self,
        thread_id: str,
        *,
        limit: int = 20,
        status: Optional[str] = None,
    ) -> list[AgentRun]:
        session = await self._get_session()
        return await run_store_list_runs_for_thread(session, thread_id, limit=limit, status=status)

    async def list_runtime_reconciliation_candidates(self, *, limit: int = 100) -> list[AgentRun]:
        """Return bounded runs whose runtime projection may need recovery.

        JSON projection metadata is intentionally filtered in Python for
        compatibility with PostgreSQL and the in-memory test session.
        """
        session = await self._get_session()
        bounded = max(1, min(int(limit), 500))
        async with session.begin():
            result = await session.execute(
                select(AgentRun)
                .where(AgentRun.status.in_([RUN_STATUS_RUNNING, RUN_STATUS_AWAITING_HUMAN, RUN_STATUS_COMPLETED, RUN_STATUS_FAILED, AgentRunStatus.CANCELLED.value]))
                .order_by(AgentRun.started_at.asc(), AgentRun.id.asc())
                .limit(max(bounded * 4, bounded))
            )
            runs = list(result.scalars().all())
        candidates: list[AgentRun] = []
        for run in runs:
            projection = dict((run.run_metadata_json or {}).get("projection") or {})
            task = await session.get(AgentTask, run.task_id) if run.task_id else None
            cancellation_pending = task is not None and str(task.status) == "cancelling"
            if cancellation_pending or projection.get("runtime_result") or projection.get("terminal_event_id") or projection.get("reconciliation_status") in {"pending", "deferred", "failed"}:
                candidates.append(run)
            if len(candidates) >= bounded:
                break
        return candidates

    async def list_nonterminal_runtime_runs(self, *, limit: int = 500) -> list[AgentRun]:
        """Return active runtime-backed runs before an intentional checkpoint reset."""
        session = await self._get_session()
        bounded = max(1, min(int(limit), 1000))
        async with session.begin():
            result = await session.execute(
                select(AgentRun)
                .where(AgentRun.status.in_([RUN_STATUS_RUNNING, RUN_STATUS_AWAITING_HUMAN]))
                .where(AgentRun.checkpoint_thread_id.is_not(None))
                .order_by(AgentRun.started_at.asc(), AgentRun.id.asc())
                .limit(bounded)
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
        return await run_store_list_chat_turns_for_run(session, run_id)

    async def create_run(
        self,
        *,
        thread_id: str,
        workflow_id: str,
        resolved_spec_json: Dict[str, Any],
        workflow_version_id: Optional[str] = None,
        workflow_version: Optional[int] = None,
        framework: Optional[str] = None,
        builder_id: Optional[str] = None,
        definition_category: Optional[str] = None,
        user_id: Optional[str] = None,
        checkpoint_thread_id: Optional[str] = None,
        runtime_binding_json: Optional[Dict[str, Any]] = None,
        run_metadata_json: Optional[Dict[str, Any]] = None,
    ) -> AgentRun:
        session = await self._get_session()
        return await run_store_create_run(
            session,
            thread_id=thread_id,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
            workflow_version=workflow_version,
            framework=framework,
            builder_id=builder_id,
            definition_category=definition_category,
            resolved_spec_json=resolved_spec_json,
            user_id=user_id,
            checkpoint_thread_id=checkpoint_thread_id,
            runtime_binding_json=runtime_binding_json,
            running_status=RUN_STATUS_RUNNING,
            run_metadata_json=run_metadata_json,
        )

    async def delete_run(self, run_id: str) -> bool:
        """Delete one exact agent run."""

        session = await self._get_session()
        return await run_store_delete_run(session, run_id)

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

        if action not in RESUME_ACTIONS and action != AgentRunResumeAction.REJECT.value:
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
            elif action == AgentRunResumeAction.REJECT.value and interrupt.get("reject_behavior") == HitlRejectBehavior.RESUME.value:
                interrupt["status"] = INTERRUPT_STATUS_RESUMED
                interrupt["decision"] = decision
                run.status = RUN_STATUS_RUNNING
                run.completed_at = None
                outcome = INTERRUPT_STATUS_RESUMED
            elif action == AgentRunResumeAction.REJECT.value:
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

    async def restore_pending_approval_after_runtime_failure(
        self,
        run_id: str,
        *,
        interrupt_id: str,
        action: str,
    ) -> bool:
        """Reopen only the exact approval decision whose runtime submission failed."""

        session = await self._get_session()
        async with session.begin():
            result = await session.execute(select(AgentRun).where(AgentRun.id == run_id).with_for_update())
            run = result.scalar_one_or_none()
            if run is None:
                return False
            interrupt = dict(run.pending_interrupt_json or {})
            if (
                run.status != RUN_STATUS_RUNNING
                or interrupt.get("status") != INTERRUPT_STATUS_RESUMED
                or interrupt.get("response_operation") != "run.approval.respond"
                or not terminal_decision_matches(interrupt, action=action, interrupt_id=interrupt_id)
            ):
                return False
            interrupt["status"] = INTERRUPT_STATUS_PENDING
            interrupt.pop("decision", None)
            run.status = RUN_STATUS_AWAITING_HUMAN
            run.completed_at = None
            metrics = dict(run.metrics_json or {})
            metrics["interrupt_resolution_count"] = max(0, int(metrics.get("interrupt_resolution_count") or 0) - 1)
            metrics.pop("last_interrupt_action", None)
            replace_jsonb_field(run, "metrics_json", metrics)
            replace_jsonb_field(run, "pending_interrupt_json", interrupt)
            return True

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
        return await run_store_complete_run(
            session,
            run_id,
            status=status,
            metrics_json=metrics_json,
            error_json=error_json,
            debug_trace_json=debug_trace_json,
            completed_at=completed_at,
        )

    async def set_run_debug_trace(
        self,
        run_id: str,
        debug_trace_json: Dict[str, Any],
    ) -> Optional[AgentRun]:
        session = await self._get_session()
        return await run_store_set_run_debug_trace(session, run_id, debug_trace_json)

    async def append_run_event(self, run_id: str, event: Any) -> bool:
        return await self.append_run_event_payload(
            run_id=run_id,
            event_id=str(getattr(event, "event_id", None) or ""),
            sequence=int(getattr(event, "sequence", 0) or 0),
            attempt=int(getattr(event, "attempt", 1) or 1),
            kind=str(getattr(event, "kind", None) or "runtime.event"),
            payload_json=dict(getattr(event, "payload", None) or {}),
            occurred_at=getattr(event, "occurred_at", None),
            trace_id=getattr(event, "trace_id", None),
            terminal=bool(getattr(event, "terminal", False)),
            source_metadata_json=dict(getattr(event, "source_metadata", None) or {}),
        )

    async def append_run_event_payload(
        self,
        *,
        run_id: str,
        event_id: str,
        kind: str,
        payload_json: Dict[str, Any],
        sequence: Optional[int] = None,
        attempt: int = 1,
        occurred_at: Any = None,
        trace_id: Optional[str] = None,
        terminal: bool = False,
        source_metadata_json: Optional[Dict[str, Any]] = None,
    ) -> bool:
        # Event sinks may persist asynchronously while the control-plane
        # request is completing. Never share the request session with those
        # writes because concurrent transactions on one session can close or
        # invalidate the transaction finalizing the run.
        session = async_session_maker()
        try:
            return await run_store_append_run_event(
                session,
                run_id=run_id,
                event_id=event_id,
                sequence=sequence,
                attempt=attempt,
                kind=kind,
                payload_json=payload_json,
                occurred_at=occurred_at,
                trace_id=trace_id,
                terminal=terminal,
                source_metadata_json=source_metadata_json,
            )
        finally:
            await session.close()

    async def list_run_events(self, run_id: str) -> list[Any]:
        session = await self._get_session()
        return await run_store_list_run_events(session, run_id)

    async def update_runtime_projection(
        self,
        run_id: str,
        projection: Dict[str, Any],
    ) -> Optional[AgentRun]:
        """Persist bounded projection/reconciliation metadata without a new table."""

        session = await self._get_session()
        async with session.begin():
            run = await session.get(AgentRun, run_id)
            if run is None:
                return None
            metadata = dict(run.run_metadata_json or {})
            metadata["projection"] = dict(projection)
            replace_jsonb_field(run, "run_metadata_json", metadata)
            return run

    async def update_run_metadata_fields(self, run_id: str, fields: Dict[str, Any]) -> Optional[AgentRun]:
        """Merge bounded control-plane metadata without replacing other owners."""

        session = async_session_maker()
        try:
            async with session.begin():
                run = await session.get(AgentRun, run_id)
                if run is None:
                    return None
                metadata = dict(run.run_metadata_json or {})
                metadata.update(dict(fields))
                replace_jsonb_field(run, "run_metadata_json", metadata)
                return run
        finally:
            await session.close()

    async def update_runtime_binding(
        self,
        run_id: str,
        binding: Any,
        *,
        status: str = "active",
    ) -> Optional[AgentRun]:
        """Persist runtime-owned opaque continuation state idempotently."""

        session = async_session_maker()
        try:
            async with session.begin():
                run = await session.get(AgentRun, run_id)
                if run is None:
                    return None
                value = binding.to_dict() if hasattr(binding, "to_dict") else dict(binding or {})
                replace_jsonb_field(run, "runtime_binding_json", value)
                run.runtime_binding_status = status
                return run
        finally:
            await session.close()

    async def mark_runtime_started(self, run_id: str) -> Optional[AgentRun]:
        """Persist that the initial runtime start has been submitted."""

        session = await self._get_session()
        async with session.begin():
            run = await session.get(AgentRun, run_id)
            if run is None:
                return None
            metadata = dict(run.run_metadata_json or {})
            metadata["runtime_started"] = True
            replace_jsonb_field(run, "run_metadata_json", metadata)
            return run


def _managed_repository_operation(method: Any) -> Any:
    """Give one public repository operation deterministic session ownership."""

    @functools.wraps(method)
    async def wrapped(self: AgentWorkflowRepository, *args: Any, **kwargs: Any) -> Any:
        if self._session is not None or self._operation_session.get() is not None:
            return await method(self, *args, **kwargs)
        async with async_session_maker() as session:
            token = self._operation_session.set(session)
            try:
                return await method(self, *args, **kwargs)
            finally:
                self._operation_session.reset(token)

    return wrapped


for _operation_name, _operation in tuple(vars(AgentWorkflowRepository).items()):
    if not _operation_name.startswith("_") and inspect.iscoroutinefunction(_operation):
        setattr(AgentWorkflowRepository, _operation_name, _managed_repository_operation(_operation))
