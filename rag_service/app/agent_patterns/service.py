from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional

from app.agent_patterns.debug_trace import AgentTraceRecorder
from app.agent_patterns.metrics import build_run_metrics
from app.agent_patterns.repository import AgentPatternRepository
from app.agent_patterns.templates import PLAN_EXECUTE_RAG_AGENT_ID, ROUTER_RAG_AGENT_ID, SUPPORTED_BUILTIN_TEMPLATE_IDS
from app.agent_patterns.validator import TemplateResolver
from app.db import get_thread_settings


logger = logging.getLogger(__name__)


class AgentRunService:
    """Runs the selected agent pattern, defaulting to the compiled Router RAG graph."""

    def __init__(
        self,
        repository: Optional[AgentPatternRepository] = None,
        resolver: Optional[TemplateResolver] = None,
    ):
        self.repository = repository or AgentPatternRepository()
        self.resolver = resolver or TemplateResolver()

    async def run_thread_chat(self, thread_id: str, req: Any, embed_model: str) -> Dict[str, Any]:
        thread_settings = await get_thread_settings(thread_id)
        agent_settings = thread_settings.get("agent_pattern") if isinstance(thread_settings, dict) else None
        agent_settings = agent_settings if isinstance(agent_settings, dict) else {}
        template_id = agent_settings.get("template_id") or ROUTER_RAG_AGENT_ID
        if template_id not in SUPPORTED_BUILTIN_TEMPLATE_IDS:
            logger.warning(
                "Unsupported agent pattern requested for thread %s | requested_template=%s fallback_template=%s",
                thread_id,
                template_id,
                ROUTER_RAG_AGENT_ID,
            )
            template_id = ROUTER_RAG_AGENT_ID
        logger.info("Resolving agent pattern for thread %s | requested_template=%s", thread_id, template_id)

        template, version = await self.repository.get_template_with_current_version(template_id)
        if template is None or version is None:
            await self.repository.seed_builtin_templates()
            template, version = await self.repository.get_template_with_current_version(template_id)
        if template is None or version is None:
            raise RuntimeError("No agent pattern is available")
        logger.info(
            "Selected agent pattern for thread %s | template=%s version=%s",
            thread_id,
            template.id,
            version.version,
        )

        request_overrides = {
            "use_web_search": getattr(req, "use_web_search", None),
            "use_reranker": getattr(req, "use_reranker", None),
            "max_iterations": getattr(req, "max_iterations", None),
            "system_role": getattr(req, "system_role_override", None),
            "tool_instructions": getattr(req, "tool_instructions_override", None),
            "custom_instructions": getattr(req, "custom_instructions_override", None),
        }
        resolved_spec = self.resolver.resolve(
            version.spec_json,
            thread_settings=thread_settings,
            request_overrides=request_overrides,
        )

        run = await self.repository.create_run(
            thread_id=thread_id,
            template_id=template.id,
            template_version_id=version.id,
            resolved_spec_json=resolved_spec,
        )

        started = time.perf_counter()
        trace_recorder = AgentTraceRecorder(run)
        context = {
            "agent_run_id": run.id,
            "agent_pattern_id": template.id,
            "agent_pattern_version": version.version,
            "agent_pattern_template_version_id": version.id,
        }

        try:
            logger.info("Invoking compiled agent pattern for thread %s | template=%s", thread_id, template.id)
            from app.agent_patterns.router_runtime import handle_plan_execute_rag_chat, handle_router_rag_chat

            handler = handle_plan_execute_rag_chat if template.id == PLAN_EXECUTE_RAG_AGENT_ID else handle_router_rag_chat
            result = await handler(
                thread_id,
                req,
                embed_model,
                resolved_spec=resolved_spec,
                agent_run_context=context,
                trace_recorder=trace_recorder,
            )
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            error_json = result.get("agent_error") if isinstance(result, dict) else None
            status = result.get("status") if isinstance(result.get("status"), str) else None
            if status is None:
                status = "failed" if error_json else "clarification" if result.get("clarification_options") else "completed"
            metrics = build_run_metrics(result, duration_ms=duration_ms)
            completed_run = await self.repository.complete_run(
                run.id,
                status=status,
                metrics_json=metrics,
                error_json=error_json,
                chat_turn_id=result.get("chat_turn_id"),
            )
            if completed_run is not None:
                debug_payload = trace_recorder.finalize(
                    run=completed_run,
                    chat_turn_id=result.get("chat_turn_id"),
                    metrics=metrics,
                    route=result.get("route"),
                    route_reason=result.get("route_reason"),
                    error=error_json,
                )
                await self.repository.set_run_debug_trace(run.id, debug_payload)
            result.update(context)
            return result
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            error_json = {
                "code": "agent_run_failed",
                "raw_message": str(exc),
                "retryable": True,
            }
            metrics = build_run_metrics({"agent_error": error_json}, duration_ms=duration_ms)
            completed_run = await self.repository.complete_run(
                run.id,
                status="failed",
                metrics_json=metrics,
                error_json=error_json,
            )
            if completed_run is not None:
                debug_payload = trace_recorder.finalize(
                    run=completed_run,
                    chat_turn_id=None,
                    metrics=metrics,
                    error=error_json,
                )
                await self.repository.set_run_debug_trace(run.id, debug_payload)
            raise
