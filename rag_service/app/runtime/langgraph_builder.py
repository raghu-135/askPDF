"""External-only LangGraph builder client.

The control plane owns product definitions and task policy. Framework validation,
materialization, catalog discovery, builder tests, and execution are delegated to
the separately deployed LangGraph runtime.
"""

from __future__ import annotations

from typing import Any, Mapping

from app.runtime.adapter import RuntimeInvocationContext
from app.runtime.builder import BuilderCapabilities, BuilderCatalog, BuilderTestContext, UnsupportedRequestOverrideError
from app.runtime.budgets import apply_deep_agent_env_overrides
from app.runtime.http_adapter import HttpLangGraphRuntimeAdapter
from runtime_protocol.contracts import AgentDefinition, AgentRuntimeRequest, AgentRuntimeResult, RuntimeValidationResult


_ALLOWED_WORKFLOW_CONFIG_KEYS = frozenset({
    "use_web_search", "use_reranker", "system_role", "tool_instructions",
    "custom_instructions", "allowed_tool_ids", "prefetch_policy", "hitl_policy",
    "replans", "graph", "context_policy", "loop_policy", "builder_ui",
    "parallel_policy", "corrective_policy", "task_policy",
})


class _BuilderEventSink:
    def __init__(self, sink: Any) -> None:
        self.sink = sink

    async def emit_runtime_event(self, event: Any) -> None:
        if self.sink is not None:
            await self.sink.emit({"event": event.kind, "data": dict(event.payload or {})})


class LangGraphBuilderProvider:
    framework = "langgraph"
    builder_id = "langgraph_graph"
    _task_web_tool_ids = frozenset({"live_web_recon"})

    def __init__(self, *, adapter: HttpLangGraphRuntimeAdapter | None = None) -> None:
        self._adapter = adapter

    def _runtime(self) -> HttpLangGraphRuntimeAdapter:
        if self._adapter is None:
            self._adapter = HttpLangGraphRuntimeAdapter()
        return self._adapter

    def supports_task_web_search(self, definition: AgentDefinition) -> bool:
        allowed = {str(value) for value in definition.definition_metadata.get("allowed_tool_ids", ()) if value}
        return bool(allowed & self._task_web_tool_ids)

    def task_configuration_fields(self, definition: AgentDefinition, spec: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        config = spec.get("config") if isinstance(spec.get("config"), Mapping) else {}
        return (
            {"id": "llm_model", "label": "Model", "type": "model", "required": True, "default": config.get("llm_model")},
            {"id": "context_window", "label": "Context window", "type": "integer", "required": True, "default": config.get("context_window"), "minimum": 1024, "read_only": False},
            {"id": "web_search_mode", "label": "Web search", "type": "enum", "required": True, "default": "off", "options": ["off", "ask", "on"], "enabled": self.supports_task_web_search(definition)},
        )

    def normalize_task_limits(self, limits: Mapping[str, Any]) -> Mapping[str, Any]:
        return apply_deep_agent_env_overrides(limits, self.framework)

    def filter_request_overrides(self, definition: AgentDefinition, overrides: Mapping[str, Any] | None, *, reject_unsupported: bool) -> Mapping[str, Any]:
        supplied = {str(key): value for key, value in dict(overrides or {}).items() if value is not None}
        unsupported = set(supplied) - _ALLOWED_WORKFLOW_CONFIG_KEYS
        if unsupported and reject_unsupported:
            raise UnsupportedRequestOverrideError(unsupported)
        return {key: value for key, value in supplied.items() if key in _ALLOWED_WORKFLOW_CONFIG_KEYS}

    async def capabilities(self, definition: AgentDefinition) -> BuilderCapabilities:
        runtime = await self._runtime().capabilities(definition)
        return BuilderCapabilities(
            framework=self.framework,
            builder_id=self.builder_id,
            authoring=True,
            transient_tests=True,
            runtime_capabilities=runtime.to_dict(),
        )

    async def validate(self, definition: AgentDefinition, spec: Mapping[str, Any], *, options: Mapping[str, Any] | None = None) -> RuntimeValidationResult:
        return await self._runtime().validate(definition, spec, options=options)

    async def normalize(self, definition: AgentDefinition, spec: Mapping[str, Any], *, options: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        return await self._runtime().resolve_definition(
            definition, spec, thread_settings={}, request_overrides={}, options=options,
        )

    async def resolve(self, definition: AgentDefinition, spec: Mapping[str, Any], *, thread_settings: Mapping[str, Any] | None = None, request_overrides: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        filtered = self.filter_request_overrides(definition, request_overrides, reject_unsupported=False)
        return await self._runtime().resolve_definition(
            definition, spec, thread_settings=thread_settings or {}, request_overrides=filtered,
        )

    async def catalog(self, definition: AgentDefinition | None = None) -> BuilderCatalog:
        target = definition or AgentDefinition("catalog", self.framework, self.builder_id)
        value = await self._runtime().builder_catalog(target)
        return BuilderCatalog(
            framework=self.framework,
            builder_id=self.builder_id,
            capabilities=await self.capabilities(target),
            payload=value,
        )

    async def source(self, definition_id: str) -> Mapping[str, Any]:
        from app.agent_workflows.builtin_workflows import load_builtin_workflows

        workflow = next((item for item in load_builtin_workflows() if item.get("builtin_key") == definition_id), None)
        if workflow is None:
            raise KeyError(definition_id)
        return {
            "builtin_key": definition_id,
            "name": workflow.get("name") or definition_id,
            "description": workflow.get("description") or "",
            "spec_json": workflow["spec_json"],
        }

    @staticmethod
    def _execution_context(context: BuilderTestContext) -> RuntimeInvocationContext:
        run = context.run
        return RuntimeInvocationContext(
            request_payload=(
                context.test_request.model_dump(mode="json")
                if hasattr(context.test_request, "model_dump")
                else dict(context.test_request or {})
                if isinstance(context.test_request, Mapping)
                else {}
            ),
            embedding_model=context.embedding_model,
            resolved_spec=dict(getattr(run, "resolved_spec_json", None) or {}),
            agent_run_context={
                "agent_run_id": str(run.id),
                "agent_workflow_id": str(run.workflow_id),
                "agent_workflow_version": int(getattr(run, "workflow_version", 1) or 1),
            },
        )

    async def transient_test(self, request: AgentRuntimeRequest, *, context: Any = None, event_sink: Any = None) -> AgentRuntimeResult:
        if not isinstance(context, BuilderTestContext):
            raise TypeError("LangGraph builder tests require BuilderTestContext")
        runtime = self._runtime()
        execution_context = self._execution_context(context)
        request = await runtime.prepare_request(request, context=execution_context)
        return await runtime.start(
            request, context=execution_context, event_sink=_BuilderEventSink(event_sink),
        )

    async def resume_transient_test(self, request: AgentRuntimeRequest, *, context: Any = None, event_sink: Any = None) -> AgentRuntimeResult:
        if not isinstance(context, BuilderTestContext):
            raise TypeError("LangGraph builder resumes require BuilderTestContext")
        runtime = self._runtime()
        execution_context = self._execution_context(context)
        request = await runtime.prepare_request(request, context=execution_context)
        return await runtime.resume(
            request,
            interrupt=dict(context.resume_decision or {}),
            context=execution_context,
            event_sink=_BuilderEventSink(event_sink),
        )

    async def cleanup_transient_test(self, request: AgentRuntimeRequest) -> Any:
        if request.continuation is None:
            return []
        return await self._runtime().delete_continuation(request.continuation)
