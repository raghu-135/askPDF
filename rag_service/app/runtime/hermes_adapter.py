"""HTTP adapter for the independent Hermes runtime service."""

from __future__ import annotations

import os
from typing import Any, Mapping

from app.runtime.contracts import AgentDefinition, AgentRuntimeRequest, AgentRuntimeResult, ContinuationBinding, RuntimeCapabilities
from app.runtime.errors import RuntimeError
from app.runtime.http_runtime_adapter import HttpRuntimeAdapter


class HermesRuntimeAdapter(HttpRuntimeAdapter):
    framework = "hermes"
    builder_id = "hermes_agent"

    def __init__(self, base_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(base_url=base_url or os.getenv("HERMES_RUNTIME_URL", "http://hermes-runtime:8200"), **kwargs)

    def _ensure_enabled(self) -> None:
        if os.getenv("HERMES_RUNTIME_ENABLED", "false").strip().lower() not in {"1", "true", "yes", "on"}:
            raise RuntimeError("runtime_disabled", "Hermes runtime is disabled")

    def _headers(self, request: AgentRuntimeRequest | None = None) -> dict[str, str]:
        headers = super()._headers(request)
        token = os.getenv("HERMES_RUNTIME_TOKEN") or os.getenv("HERMES_API_TOKEN")
        if token:
            headers["authorization"] = f"Bearer {token}"
        elif os.getenv("LANGGRAPH_RUNTIME_TOKEN") and headers.get("authorization") == f"Bearer {os.getenv('LANGGRAPH_RUNTIME_TOKEN')}":
            headers.pop("authorization", None)
        return headers

    async def start(self, request: AgentRuntimeRequest, *, context: Any, event_sink: Any = None) -> AgentRuntimeResult:
        self._ensure_enabled()
        return await super().start(request, context=context, event_sink=event_sink)

    async def capabilities(self, definition: AgentDefinition) -> RuntimeCapabilities:
        value = await self._json("GET", "/v1/capabilities")
        from app.runtime.transport import capabilities_from_dict
        return capabilities_from_dict(value.get("capabilities") or value)

    async def resume(self, request: AgentRuntimeRequest, *, interrupt: Mapping[str, Any], context: Any, event_sink: Any = None) -> AgentRuntimeResult:
        raise RuntimeError("runtime_capability_unsupported", "Hermes resume is not enabled for this proof runtime")

    async def continue_run(self, request: AgentRuntimeRequest, *, context: Any, event_sink: Any = None) -> AgentRuntimeResult | None:
        raise RuntimeError("runtime_capability_unsupported", "Hermes continuation is not enabled for this proof runtime")

    async def cancel(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        self._ensure_enabled()
        if request.continuation is None or not request.continuation.payload.get("upstream_run_id"):
            raise RuntimeError("runtime_binding_missing", "Hermes cancellation requires an upstream run binding")
        value = await self._json(
            "POST",
            f"/v1/runs/{request.run_id}/cancel",
            request=request,
            json={"request": request.to_dict(), "continuation": request.continuation.to_dict()},
        )
        return dict(value or {})

    async def inspect(self, request: AgentRuntimeRequest) -> Mapping[str, Any]:
        self._ensure_enabled()
        if request.continuation is None or not request.continuation.payload.get("upstream_run_id"):
            raise RuntimeError("runtime_binding_missing", "Hermes inspection requires an upstream run binding")
        value = await self._json(
            "POST",
            f"/v1/runs/{request.run_id}/inspect",
            request=request,
            json={"request": request.to_dict(), "continuation": request.continuation.to_dict()},
        )
        return dict(value or {})

    async def delete_continuation(self, continuation: ContinuationBinding) -> Any:
        raise RuntimeError("runtime_capability_unsupported", "Hermes does not expose safe durable session deletion")
