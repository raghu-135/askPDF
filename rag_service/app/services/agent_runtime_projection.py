"""Control-plane projection of persistence-free runtime results."""

from __future__ import annotations

from typing import Any, Mapping

from app.db import ChatTurnStatus, ReasoningFormat, create_chat_turn, increment_qa_stats, update_message_context_compact
from app.rag.indexer import index_chat_memory_for_thread


class AgentRuntimeProjection:
    """Persist canonical product records after runtime execution completes."""

    async def apply_event(self, *, run: Any, event: Any) -> bool:
        """Apply only events newer than the persisted projection sequence."""

        from app.agent_workflows.repository import AgentWorkflowRepository

        event_id = str(getattr(event, "event_id", None) or "")
        sequence = int(getattr(event, "sequence", 0) or 0)
        return await AgentWorkflowRepository().apply_runtime_projection_event(
            run.id,
            event_id=event_id,
            sequence=sequence,
            continuation=getattr(event, "continuation", None),
            checkpoint_boundary_available=getattr(event, "checkpoint_boundary_available", None),
        )

    async def project_terminal_result(self, *, run: Any, result: Mapping[str, Any], terminal_event_id: str | None = None) -> dict[str, Any]:
        from app.services.agent_runtime_reconciliation import record_terminal_result

        await record_terminal_result(run, result, terminal_event_id=terminal_event_id)
        return await self.reconcile_run(run=run, result=result)

    async def persist_trace(self, *, run: Any, events: list[Any]) -> None:
        for event in events:
            await self.apply_event(run=run, event=event)

    async def rebuild_trace_from_events(self, *, run: Any, result: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        """Rebuild the product trace from the canonical runtime journal."""

        from app.agent_workflows.debug_trace import AgentTraceRecorder, finalize_and_merge_debug_payload
        from app.agent_workflows.repository import AgentWorkflowRepository
        from app.runtime.contracts import AgentRuntimeEvent

        repository = AgentWorkflowRepository()
        events = await repository.list_run_events(run.id)
        if not events:
            return None
        recorder = AgentTraceRecorder(run)
        for event in events:
            recorder.record_agent_runtime_event(AgentRuntimeEvent(
                event_id=str(event.event_id),
                run_id=str(event.agent_run_id),
                sequence=int(event.sequence),
                attempt=int(event.attempt),
                kind=str(event.kind),
                payload=event.payload_json if isinstance(event.payload_json, dict) else {},
                occurred_at=str(event.occurred_at) if event.occurred_at else None,
                terminal=bool(event.terminal),
                source_metadata=event.source_metadata_json if isinstance(event.source_metadata_json, dict) else {},
            ))
        result_payload = dict(result or {})
        debug = finalize_and_merge_debug_payload(
            recorder=recorder,
            run=run,
            metrics=dict(getattr(run, "metrics_json", None) or {}),
            result=result_payload or None,
            chat_turn_id=result_payload.get("chat_turn_id"),
            route=result_payload.get("route"),
            route_reason=result_payload.get("route_reason"),
            error=result_payload.get("agent_error") or getattr(run, "error_json", None),
            run_status=str(result_payload.get("status") or getattr(run, "status", "")),
            completed_at=getattr(run, "completed_at", None),
        )
        await repository.set_run_debug_trace(run.id, debug)
        return debug

    async def project_chat_result(
        self,
        *,
        thread_id: str,
        question: str,
        result: Mapping[str, Any],
        run_context: Mapping[str, Any] | None = None,
        duration_ms: float = 0.0,
        success_context: str = "",
        agent_run_context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        projected = dict(result)
        run_context = run_context or agent_run_context or {}
        status = str(projected.get("status") or "completed")
        if status not in {ChatTurnStatus.COMPLETED.value, "failed"}:
            return projected
        from app.services.agent_runtime_reconciliation import result_hash

        run_id = run_context.get("agent_run_id")
        digest = result_hash(projected)
        existing = None
        repository = None
        projection: dict[str, Any] = {}
        if run_id:
            from app.agent_workflows.repository import AgentWorkflowRepository

            repository = AgentWorkflowRepository()
            fresh_run = await repository.get_run(str(run_id))
            projection = dict(
                ((getattr(fresh_run, "run_metadata_json", None) or {}).get("projection") or {})
            )
            applied_hash = projection.get("result_hash")
            if projection.get("status") == "applied":
                if applied_hash != digest:
                    raise ValueError("runtime_terminal_result_conflict")
                projected.update(
                    {
                        "answer": projected.get("final_answer") or projected.get("answer") or "",
                        "chat_turn_id": projection.get("chat_turn_id"),
                        "user_message_id": projection.get("user_message_id"),
                        "assistant_message_id": projection.get("assistant_message_id"),
                        "agent_run_turn_kind": "assistant_final",
                        "agent_run_sequence": 0,
                        "duration_ms": duration_ms,
                    }
                )
                return projected
            if applied_hash and applied_hash != digest:
                raise ValueError("runtime_terminal_result_conflict")

            turns = await repository.list_chat_turns_for_run(str(run_id))
            existing = next(
                (
                    turn for turn in turns
                    if turn.agent_run_turn_kind == "assistant_final" and turn.agent_run_sequence == 0
                ),
                None,
            )
        if existing is None:
            answer = str(
                projected.get("final_answer")
                or projected.get("answer")
                or "I'm sorry, I encountered a technical error while processing your request."
            )
            turn = await create_chat_turn(
                thread_id=thread_id,
                question=question,
                answer=answer,
                rewritten_question=None,
                status=ChatTurnStatus.COMPLETED.value if status == ChatTurnStatus.COMPLETED.value else "failed",
                reasoning=projected.get("reasoning") or "",
                reasoning_available=bool(projected.get("reasoning_available")),
                reasoning_format=projected.get("reasoning_format") or ReasoningFormat.NONE.value,
                web_sources=projected.get("web_sources") or [],
                document_sources=projected.get("document_sources") or [],
                used_chat_ids=projected.get("used_chat_ids") or [],
                clarification_options=None,
                metadata={
                    "agent_workflow_id": run_context.get("agent_workflow_id"),
                    "agent_route": projected.get("route"),
                    "agent_route_reason": projected.get("route_reason"),
                },
                agent_run_id=run_id,
                agent_run_turn_kind="assistant_final",
                agent_run_sequence=0,
                agent_trace_refs_json=projected.get("agent_trace_refs"),
            )
        else:
            turn = existing
        if projected.get("embedding_model") and projected.get("llm_model"):
            indexed = await index_chat_memory_for_thread(
                thread_id=thread_id,
                message_id=turn.id,
                question=question,
                answer=str(projected.get("final_answer") or projected.get("answer") or ""),
                embedding_model=projected["embedding_model"],
                llm_name=projected["llm_model"],
                context_window=projected.get("context_window"),
                message_created_at=turn.completed_at or turn.created_at,
            )
            compact = indexed.get("memory_compact_text") if isinstance(indexed, dict) else None
            if compact:
                await update_message_context_compact(turn.id, compact)
        try:
            await increment_qa_stats(thread_id, len(question or "") + len(str(projected.get("answer") or projected.get("final_answer") or "")))
        except Exception:
            pass
        projected.update(
            {
                "answer": projected.get("final_answer") or projected.get("answer") or "",
                "chat_turn_id": turn.id,
                "user_message_id": f"{turn.id}:user",
                "assistant_message_id": f"{turn.id}:assistant",
                "agent_run_turn_kind": "assistant_final",
                "agent_run_sequence": 0,
                "duration_ms": duration_ms,
            }
        )
        if run_id and repository is not None:
            if hasattr(repository, "update_runtime_projection"):
                await repository.update_runtime_projection(
                    str(run_id),
                    {
                        **projection,
                        "status": "applied",
                        "result_hash": digest,
                        "terminal_event_id": (
                            projected.get("terminal_event_id")
                            or projection.get("terminal_event_id")
                        ),
                        "chat_turn_id": turn.id,
                        "user_message_id": f"{turn.id}:user",
                        "assistant_message_id": f"{turn.id}:assistant",
                    },
                )
        return projected

    async def reconcile_run(self, *, run: Any, result: Mapping[str, Any] | None = None) -> Any:
        """Reapply a known terminal result without recreating paused runs."""

        if result is None:
            return run
        if str(result.get("status") or "") in {"awaiting_human", "paused"}:
            return run
        projected = await self.project_chat_result(
            thread_id=run.thread_id,
            question=str(result.get("question") or ""),
            result=result,
            run_context={"agent_run_id": run.id, "agent_workflow_id": run.workflow_id},
            duration_ms=float(result.get("duration_ms") or 0),
        )
        await self.rebuild_trace_from_events(run=run, result=projected)
        return projected
