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

    async def delete_continuation(self, continuation: ContinuationBinding) -> Any:
        if continuation is None:
            return {"status": "empty"}
        session_id = str(continuation.payload.get("session_id") or "")
        return await self._json("DELETE", f"/v1/continuations/{session_id}", json={"continuation": continuation.to_dict()})
