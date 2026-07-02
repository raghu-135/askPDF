from __future__ import annotations

import time
import logging
from typing import Any, Dict, Optional

from app.agent_patterns.repository import AgentPatternRepository
from app.agent_patterns.templates import ROUTER_RAG_AGENT_ID
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
        if template_id != ROUTER_RAG_AGENT_ID:
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
            "use_intent_agent": getattr(req, "use_intent_agent", None),
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
        context = {
            "agent_run_id": run.id,
            "agent_pattern_id": template.id,
            "agent_pattern_version": version.version,
            "agent_pattern_template_version_id": version.id,
        }

        try:
            logger.info("Invoking compiled Router RAG Agent for thread %s", thread_id)
            from app.agent_patterns.router_runtime import handle_router_rag_chat

            result = await handle_router_rag_chat(
                thread_id,
                req,
                embed_model,
                resolved_spec=resolved_spec,
                agent_run_context=context,
            )
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            error_json = result.get("agent_error") if isinstance(result, dict) else None
            status = "failed" if error_json else "completed"
            metrics = {
                "duration_ms": duration_ms,
                "document_source_count": len(result.get("document_sources") or []),
                "web_source_count": len(result.get("web_sources") or []),
                "used_chat_id_count": len(result.get("used_chat_ids") or []),
                "clarification": bool(result.get("clarification_options")),
                "route": result.get("route"),
                "node_event_count": len(result.get("node_events") or []),
            }
            await self.repository.complete_run(
                run.id,
                status=status,
                metrics_json=metrics,
                error_json=error_json,
            )
            result.update(context)
            return result
        except Exception as exc:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            await self.repository.complete_run(
                run.id,
                status="failed",
                metrics_json={"duration_ms": duration_ms},
                error_json={
                    "code": "agent_run_failed",
                    "raw_message": str(exc),
                    "retryable": True,
                },
            )
            raise
