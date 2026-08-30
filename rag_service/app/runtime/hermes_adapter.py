"""HTTP adapter for the independent Hermes runtime service."""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any, Mapping

from app.runtime.contracts import (
    AgentDefinition,
    AgentRuntimeRequest,
    AgentRuntimeResult,
    RuntimeApprovalResponse,
    RuntimeCapabilities,
    RuntimeCapabilityDisabledReason,
    RuntimeCapabilitySemantics,
    RuntimeFeatureId,
    RuntimeFeatureDescriptor,
    RuntimeOperationId,
    RuntimeSupportLevel,
    RuntimeValidationResult,
)
from app.runtime.errors import RuntimeError
from app.runtime.adapter import AgentRuntimeAdapter
from app.runtime.http_runtime_adapter import RuntimeTransportConnector
from app.runtime.hermes_config import HermesConfigurationError, hermes_runtime_enabled, validate_hermes_model_compatibility
from app.models.llm_server_client import check_model_can_invoke_tools
from app.mcp.execution_context_token import issue_execution_context_token
from app.tools.context import ToolInvocationContext


class HermesRuntimeAdapter(AgentRuntimeAdapter):
    framework = "hermes"
    builder_id = "hermes_agent"
    # The pinned Hermes HTTP Runs API has no durable pause/checkpoint
    # primitive. Keep task pause out of its effective capabilities.
    supports_task_pause = False
    visualization_id = "hermes.session"
    implemented_operations = frozenset({
        RuntimeOperationId.RUN_START,
        RuntimeOperationId.RUN_CANCEL,
        RuntimeOperationId.RUN_APPROVAL_RESPOND,
        RuntimeOperationId.RUN_INSPECT_STATE,
        RuntimeOperationId.TRACE_PROJECT,
    })

    async def prepare_request(
        self,
        request: AgentRuntimeRequest,
        *,
        context: Any,
    ) -> AgentRuntimeRequest:
        task_context = getattr(context, "task_context", None)
        if task_context is None:
            return request
        data = dict(getattr(task_context, "context_data", {}) or {})
        limits = dict(getattr(task_context, "limits", {}) or {})
        spec = dict(getattr(context, "resolved_spec", {}) or {})
        config = dict(spec.get("config") or {})
        profile = dict(spec.get("managed_profile") or {})
        mcp = dict(profile.get("mcp") or config.get("mcp") or {})
        allowed_tools = list(mcp.get("allowed_tool_ids") or config.get("allowed_tool_ids") or [])
        ttl_seconds = max(3600, int(limits.get("max_active_runtime_ms", 3_600_000)) // 1000)
        context_window = int(
            profile.get("context_window")
            or config.get("context_window")
            or data.get("context_window")
            or 32_768
        )
        token = issue_execution_context_token(
            ToolInvocationContext(
                thread_id=request.thread_id,
                run_id=request.run_id,
                embedding_model=context.embedding_model,
                context_window=context_window,
                use_web_search=bool(config.get("use_web_search")),
                use_reranker=True,
                extensions={"task_id": task_context.task_id, "llm_model": config.get("llm_model")},
            ),
            task_id=task_context.task_id,
            allowed_tools=allowed_tools,
            ttl_seconds=ttl_seconds,
        )
        return replace(request, input={
            **dict(request.input),
            "task_context": data,
            "mcp_execution_context_token": token,
        })

    def __init__(self, base_url: str | None = None, **kwargs: Any) -> None:
        configured_base_url = base_url or os.getenv("HERMES_RUNTIME_URL", "").strip()
        if not configured_base_url:
            raise RuntimeError("runtime_configuration_invalid", "HERMES_RUNTIME_URL is required for the Hermes runtime")
        self.transport = RuntimeTransportConnector(
            base_url=configured_base_url,
            framework=self.framework,
            authorization_envs=("HERMES_RUNTIME_TOKEN", "HERMES_API_TOKEN"),
            visualization_id=self.visualization_id,
            replay_by_event_id=True,
            **kwargs,
        )

    def _ensure_enabled(self) -> None:
        if not hermes_runtime_enabled():
            raise RuntimeError("runtime_disabled", "Hermes runtime is disabled")
        try:
            validate_hermes_model_compatibility()
        except HermesConfigurationError as exc:
            raise RuntimeError("runtime_configuration_invalid", str(exc)) from exc

    async def start(self, request: AgentRuntimeRequest, *, context: Any, event_sink: Any = None) -> AgentRuntimeResult:
        self._ensure_enabled()
        resolved_spec = dict(getattr(context, "resolved_spec", None) or {})
        model = str(((resolved_spec.get("managed_profile") or {}).get("model_policy") or {}).get("model") or "").strip()
        if not model or not await check_model_can_invoke_tools(model):
            raise RuntimeError(
                "runtime_model_tool_calling_unsupported",
                "The selected model cannot invoke the tools required by Hermes",
                details={"framework": self.framework, "model": model or None},
            )
        return await self.transport._stream("/v1/runs/start", request, context=context, payload=None, event_sink=event_sink)

    async def capabilities(self, definition: AgentDefinition) -> RuntimeCapabilities:
        capabilities = await self.deployment_capabilities()
        policy = definition.definition_metadata.get("runtime_policy") or definition.definition_metadata.get("hermes_policy")
        if not isinstance(policy, Mapping):
            return capabilities

        def feature(enabled: bool, semantics: RuntimeCapabilitySemantics, details: Mapping[str, Any]) -> RuntimeFeatureDescriptor:
            return RuntimeFeatureDescriptor(
                support=RuntimeSupportLevel.NATIVE if enabled else RuntimeSupportLevel.UNSUPPORTED,
                enabled=enabled,
                disabled_reason=(
                    None
                    if enabled
                    else RuntimeCapabilityDisabledReason.DEFINITION_POLICY
                ),
                semantics=semantics,
                details=dict(details),
            )

        features = dict(capabilities.features)
        allowed_tools = tuple(str(item) for item in policy.get("allowed_tool_ids", ()) if item)
        features.update({
            RuntimeFeatureId.TOOLS: feature(
                bool(allowed_tools),
                RuntimeCapabilitySemantics.DEFINITION_TOOL_POLICY,
                {"allowed_tool_ids": list(allowed_tools)},
            ),
            RuntimeFeatureId.MEMORY: feature(
                bool(policy.get("allow_persistent_memory")),
                RuntimeCapabilitySemantics.DEFINITION_TOOL_POLICY,
                {"persistent": bool(policy.get("allow_persistent_memory"))},
            ),
            RuntimeFeatureId.DELEGATION: feature(
                bool(policy.get("allow_subagents")),
                RuntimeCapabilitySemantics.PRODUCT_MANAGED_SUBAGENTS,
                {"enabled": bool(policy.get("allow_subagents"))},
            ),
            RuntimeFeatureId.SKILLS: feature(
                bool(policy.get("skills")),
                RuntimeCapabilitySemantics.DEFINITION_TOOL_POLICY,
                {"skills": list(policy.get("skills", ()))},
            ),
        })
        operations = dict(capabilities.operations)
        approval = operations.get(RuntimeOperationId.RUN_APPROVAL_RESPOND)
        if approval is not None and not bool(policy.get("approval_enabled", True)):
            operations[RuntimeOperationId.RUN_APPROVAL_RESPOND] = replace(
                approval,
                enabled=False,
                disabled_reason=RuntimeCapabilityDisabledReason.DEFINITION_POLICY,
            )
        return replace(capabilities, operations=operations, features=features)

    async def deployment_capabilities(self) -> RuntimeCapabilities:
        self._ensure_enabled()
        value = await self.transport._json("GET", "/v1/capabilities")
        from app.runtime.transport import capabilities_from_dict
        try:
            capabilities = value["capabilities"]
            if not isinstance(capabilities, Mapping):
                raise ValueError("capabilities must be an object")
            return capabilities_from_dict(capabilities)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("runtime_protocol_error", "Agent runtime returned malformed capabilities") from exc

    async def validate(
        self,
        definition: AgentDefinition,
        spec: Mapping[str, Any],
        *,
        options: Mapping[str, Any] | None = None,
    ) -> RuntimeValidationResult:
        value = await self.transport._json(
            "POST",
            "/v1/validate",
            json={
                "definition": definition.to_dict(),
                "spec": dict(spec),
                "options": dict(options or {}),
            },
        )
        from app.runtime.transport import validation_from_dict
        return validation_from_dict(value["validation"])

    async def resume(self, request: AgentRuntimeRequest, *, interrupt: Mapping[str, Any], context: Any, event_sink: Any = None) -> AgentRuntimeResult:
        self._unsupported("run.resume", "Hermes resume is not supported by the pinned runs API")

    async def continue_run(self, request: AgentRuntimeRequest, *, context: Any, event_sink: Any = None) -> AgentRuntimeResult | None:
        self._ensure_enabled()
        if request.continuation is None or not request.continuation.payload.get("upstream_run_id"):
            raise RuntimeError("runtime_binding_missing", "Hermes continuation requires an upstream run binding")
        result = await self.transport._stream(f"/v1/runs/{request.run_id}/continue", request, context=context, payload=None, event_sink=event_sink)
        return None if result.status == "no_continuation" else result

    async def cancel(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        self._ensure_enabled()
        if request.continuation is None or not request.continuation.payload.get("upstream_run_id"):
            raise RuntimeError("runtime_binding_missing", "Hermes cancellation requires an upstream run binding")
        value = await self.transport._json(
            "POST",
            f"/v1/runs/{request.run_id}/cancel",
            request=request,
            json={"request": request.to_dict(), "continuation": request.continuation.to_dict()},
        )
        return dict(value or {})

    async def respond_to_approval(self, request: AgentRuntimeRequest, response: RuntimeApprovalResponse) -> Mapping[str, Any]:
        self._ensure_enabled()
        if request.continuation is None or not request.continuation.payload.get("upstream_run_id"):
            raise RuntimeError("runtime_binding_missing", "Hermes approval requires an upstream run binding")
        if response.decision not in {"approve", "reject"}:
            raise RuntimeError("invalid_approval_response", "Hermes approvals support approve or reject")
        choice = response.scope if response.decision == "approve" else "deny"
        if choice not in {"once", "session", "always", "deny"}:
            raise RuntimeError("invalid_approval_response", "Hermes approval scope is invalid")
        value = await self.transport._json(
            "POST",
            f"/v1/runs/{request.run_id}/approval",
            request=request,
            json={
                "request": request.to_dict(),
                "continuation": request.continuation.to_dict(),
                "response": {"choice": choice, "resolve_all": choice in {"session", "always"}},
            },
        )
        return dict(value or {})

    async def inspect_state(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        self._ensure_enabled()
        if request.continuation is None or not request.continuation.payload.get("upstream_run_id"):
            raise RuntimeError("runtime_binding_missing", "Hermes inspection requires an upstream run binding")
        value = await self.transport._json(
            "POST",
            f"/v1/runs/{request.run_id}/inspect",
            request=request,
            json={"request": request.to_dict(), "continuation": request.continuation.to_dict()},
        )
        return dict(value or {})
