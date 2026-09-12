"""LangGraph adapter for the separately deployable runtime service."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from app.runtime.adapter import AgentRuntimeAdapter, AgentRuntimeEventSink, RuntimeInvocationContext
from runtime_protocol.contracts import (
    AgentDefinition,
    AgentRuntimeEvent,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    RuntimeCleanupResult,
    RuntimeCapabilities,
    RuntimeCourseCorrection,
    RuntimeCourseCorrectionReceipt,
    RuntimeOperationId,
    RuntimeValidationResult,
)
from runtime_protocol.errors import RuntimeError
from runtime_protocol.transport import (
    capabilities_from_dict,
    course_correction_receipt_from_dict,
    event_from_dict,
    validation_from_dict,
)
from runtime_protocol.protocol import json_payload
from app.runtime.http_transport import RuntimeTransportConnector, context_to_dict, safe_json


class HttpLangGraphRuntimeAdapter(AgentRuntimeAdapter):
    """LangGraph adapter composed with the neutral HTTP transport."""

    # The external runtime exposes a durable pause request protocol backed by
    # its execution store and LangGraph checkpointer.
    framework = "langgraph"
    builder_id = "langgraph_graph"
    implemented_operations = frozenset({
        RuntimeOperationId.RUN_START,
        RuntimeOperationId.RUN_CANCEL,
        RuntimeOperationId.RUN_RESUME,
        RuntimeOperationId.RUN_INSPECT_STATE,
        RuntimeOperationId.RUN_CLEANUP,
        RuntimeOperationId.TASK_PAUSE,
        RuntimeOperationId.TASK_COURSE_CORRECTION_SUBMIT,
        RuntimeOperationId.TRACE_PROJECT,
    })

    def __init__(self, base_url: str | None = None, **kwargs: Any) -> None:
        self.transport = RuntimeTransportConnector(
            base_url=base_url,
            framework=self.framework,
            authorization_env="LANGGRAPH_RUNTIME_TOKEN",
            visualization_id="langgraph.session",
            **kwargs,
        )

    async def prepare_request(
        self,
        request: AgentRuntimeRequest,
        *,
        context: RuntimeInvocationContext,
    ) -> AgentRuntimeRequest:
        from app.mcp.execution_context_token import execution_context_ttl_seconds, issue_execution_context_token
        from app.mcp.registry import MCP_TOOL_DEFINITIONS
        from app.tools.context import ToolInvocationContext

        spec = dict(context.resolved_spec or {})
        config = dict(spec.get("config") or {})
        use_reranker = config.get("use_reranker")
        if not isinstance(use_reranker, bool):
            raise RuntimeError(
                "runtime_configuration_invalid",
                "Resolved workflow configuration must provide boolean use_reranker",
                retryable=False,
            )
        configured_tool_ids = {
            str(value) for value in config.get("allowed_tool_ids") or [] if value
        }
        # Workflow specs expose stable, framework-neutral contract IDs, while
        # the MCP authorization boundary validates canonical MCP tool names.
        # Expand the grant here, where the control plane owns both registries,
        # instead of teaching the external runtime about product policy.
        allowed_tools = sorted(
            name
            for name, definition in MCP_TOOL_DEFINITIONS.items()
            if name in configured_tool_ids
            or definition.registry_contract_id in configured_tool_ids
        )
        task_context = context.task_context
        task_id = str(request.task_id or request.run_id)
        limits = dict(task_context.limits or {}) if task_context is not None else {}
        ttl_seconds = execution_context_ttl_seconds(limits)
        token = issue_execution_context_token(
            ToolInvocationContext(
                thread_id=request.thread_id,
                run_id=request.run_id,
                embedding_model=context.embedding_model,
                context_window=int(config.get("context_window") or 32_768),
                use_web_search=bool(config.get("use_web_search")),
                use_reranker=use_reranker,
                extensions={"task_id": task_id, "llm_model": config.get("llm_model")},
            ),
            task_id=task_id,
            allowed_tools=allowed_tools,
            ttl_seconds=ttl_seconds,
            runtime="langgraph",
        )
        return replace(
            request,
            input={
                **dict(request.input),
                "mcp_execution_context_token": token,
                # Admission must check the canonical MCP grant, not every
                # framework-neutral tool contract in the workflow. Some
                # contracts (for example ``clarify_intent``) are implemented
                # inside the graph and are intentionally absent from MCP.
                "mcp_allowed_tool_ids": allowed_tools,
            },
        )

    async def aclose(self) -> None:
        await self.transport.aclose()

    async def capabilities(self, definition: AgentDefinition) -> RuntimeCapabilities:
        value = await self.transport._json("POST", "/v1/capabilities", json=json_payload({"definition": definition.to_dict()}))
        return capabilities_from_dict(value["capabilities"])

    async def deployment_capabilities(self) -> RuntimeCapabilities:
        value = await self.transport._json("GET", "/v1/capabilities")
        return capabilities_from_dict(value["capabilities"])

    async def readiness(self) -> Mapping[str, Any]:
        return await self.transport._readiness()

    async def startup_readiness(self) -> Mapping[str, Any]:
        """Probe runtime process/core initialization without its CP callback.

        The runtime's full ``/readyz`` also checks the control-plane MCP
        callback.  Using it during control-plane startup would deadlock the
        two services; request admission and the ongoing readiness loop still
        use the full readiness endpoint.
        """
        return await self.transport._readiness("/startupz")

    async def validate(self, definition: AgentDefinition, spec: Mapping[str, Any], *, options: Mapping[str, Any] | None = None) -> RuntimeValidationResult:
        value = await self.transport._json("POST", "/v1/validate", json=json_payload({"definition": definition.to_dict(), "spec": safe_json(spec), "options": safe_json(options or {})}))
        return validation_from_dict(value["validation"])

    async def resolve_definition(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        thread_settings: Mapping[str, Any],
        request_overrides: Mapping[str, Any],
        options: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        value = await self.transport._json(
            "POST",
            "/v1/resolve",
            json=json_payload({
                "definition": definition.to_dict(),
                "spec": safe_json(spec),
                "thread_settings": safe_json(thread_settings),
                "request_overrides": safe_json(request_overrides),
                "options": safe_json(options or {}),
            }),
        )
        resolved = value.get("resolved_spec")
        if not isinstance(resolved, Mapping):
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned an invalid resolved definition")
        return dict(resolved)

    async def builder_catalog(self, definition: AgentDefinition) -> Mapping[str, Any]:
        value = await self.transport._json(
            "POST", "/v1/catalog", json=json_payload({"definition": definition.to_dict()})
        )
        catalog = value.get("catalog")
        if not isinstance(catalog, Mapping):
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned an invalid builder catalog")
        return dict(catalog)

    async def prompt_preview(self, definition: AgentDefinition, spec: Mapping[str, Any], options: Mapping[str, Any]) -> str:
        value = await self.transport._json(
            "POST", "/v1/prompt-preview",
            json=json_payload({"definition": definition.to_dict(), "spec": safe_json(spec), "options": safe_json(options)}),
        )
        prompt = value.get("prompt")
        if not isinstance(prompt, str):
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned an invalid prompt preview")
        return prompt

    async def start(self, request: AgentRuntimeRequest, *, context: RuntimeInvocationContext, event_sink: AgentRuntimeEventSink | None = None) -> AgentRuntimeResult:
        return await self.transport._stream("/v1/runs/start", request, context=context, payload=None, event_sink=event_sink)

    async def resume(self, request: AgentRuntimeRequest, *, interrupt: Mapping[str, Any], context: RuntimeInvocationContext, event_sink: AgentRuntimeEventSink | None = None) -> AgentRuntimeResult:
        return await self.transport._stream(f"/v1/runs/{request.run_id}/resume", request, context=context, payload={"interrupt": safe_json(interrupt)}, event_sink=event_sink)

    async def continue_run(self, request: AgentRuntimeRequest, *, context: RuntimeInvocationContext, event_sink: AgentRuntimeEventSink | None = None) -> AgentRuntimeResult | None:
        result = await self.transport._stream(
            f"/v1/runs/{request.run_id}/continue",
            request,
            context=context,
            payload=None,
            event_sink=event_sink,
        )
        return None if result.status == "no_continuation" else result

    async def cancel(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        value = await self.transport._json("POST", f"/v1/runs/{request.run_id}/cancel", request=request, json=json_payload({"request": request.to_dict()}))
        return dict(value) if isinstance(value, Mapping) else {"result": value}

    async def pause(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        value = await self.transport._json("POST", f"/v1/runs/{request.run_id}/pause", request=request, json=json_payload({"request": request.to_dict()}))
        return dict(value) if isinstance(value, Mapping) else {"result": value}

    async def submit_course_correction(
        self,
        request: AgentRuntimeRequest,
        correction: RuntimeCourseCorrection,
    ) -> RuntimeCourseCorrectionReceipt:
        value = await self.transport._json(
            "POST",
            f"/v1/runs/{request.run_id}/course-corrections",
            request=request,
            json=json_payload({"request": request.to_dict(), "correction": correction.to_dict()}),
        )
        if not isinstance(value, Mapping):
            raise RuntimeError("runtime_protocol_error", "Runtime returned an invalid correction receipt")
        return course_correction_receipt_from_dict(value)

    async def inspect_state(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        value = await self.transport._json("POST", f"/v1/runs/{request.run_id}/inspect", request=request, json=json_payload({"request": request.to_dict()}))
        return dict(value or {}) if isinstance(value, Mapping) else {}

    async def project_trace(self, events: list[Mapping[str, Any]], *, run_id: str, context: RuntimeInvocationContext | None = None) -> list[AgentRuntimeEvent]:
        projected = []
        for event in events:
            value = dict(event)
            source_metadata = dict(value.get("source_metadata") or {})
            source_metadata.setdefault("framework", self.framework)
            source_metadata.setdefault("source_event", str(value.get("kind") or "runtime.event"))
            source_metadata.setdefault("visualization_id", "langgraph.session")
            value["source_metadata"] = source_metadata
            projected.append(event_from_dict(value))
        return projected

    async def cleanup_run(self, run_id: str) -> Any:
        value = await self.transport._json("DELETE", f"/v1/runs/{run_id}", json={})
        try:
            return RuntimeCleanupResult.from_mapping(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned an invalid cleanup result") from exc
