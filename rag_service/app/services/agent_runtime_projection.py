"""Control-plane projection of persistence-free runtime results."""

from __future__ import annotations

from typing import Any, Mapping
import hashlib
import json

from app.db import ChatTurnStatus, ReasoningFormat, create_chat_turn, increment_qa_stats, update_message_context_compact
from app.rag.indexer import index_chat_memory_for_thread


class AgentRuntimeProjection:
    """Persist canonical product records after runtime execution completes."""

    async def apply_event(self, *, run: Any, event: Any) -> bool:
        """Record the bounded last event marker; duplicate IDs are ignored."""

        from app.agent_workflows.repository import AgentWorkflowRepository

        metadata = dict(getattr(run, "run_metadata_json", None) or {})
        projection = dict(metadata.get("projection") or {})
        event_id = str(getattr(event, "event_id", None) or "")
        if event_id and event_id == projection.get("last_event_id"):
            return False
        projection.update(
            {
                "version": 1,
                "status": projection.get("status") or "pending",
                "last_event_id": event_id,
                "last_event_sequence": int(getattr(event, "sequence", 0) or 0),
            }
        )
        await AgentWorkflowRepository().update_runtime_projection(run.id, projection)
        continuation = getattr(event, "continuation", None)
        if continuation is not None:
            await AgentWorkflowRepository().update_runtime_binding(run.id, continuation)
        return True

    async def project_terminal_result(self, *, run: Any, result: Mapping[str, Any], terminal_event_id: str | None = None) -> dict[str, Any]:
        from app.services.agent_runtime_reconciliation import record_terminal_result

        await record_terminal_result(run, result, terminal_event_id=terminal_event_id)
        return await self.reconcile_run(run=run, result=result)

    async def persist_trace(self, *, run: Any, events: list[Any]) -> None:
        for event in events:
            await self.apply_event(run=run, event=event)

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
        run_id = run_context.get("agent_run_id")
        result_hash = hashlib.sha256(
            json.dumps(projected, sort_keys=True, default=str, separators=(",", ":")).encode()
        ).hexdigest()
        existing = None
        if run_id:
            from app.agent_workflows.repository import AgentWorkflowRepository

            turns = await AgentWorkflowRepository().list_chat_turns_for_run(str(run_id))
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
        if run_id:
            from app.agent_workflows.repository import AgentWorkflowRepository

            repository = AgentWorkflowRepository()
            if hasattr(repository, "update_runtime_projection"):
                await repository.update_runtime_projection(
                    str(run_id),
                    {
                        "version": 1,
                        "status": "applied",
                        "result_hash": result_hash,
                        "terminal_event_id": projected.get("terminal_event_id"),
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
        return projected
