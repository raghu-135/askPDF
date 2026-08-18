"""Runtime adapter registry keyed by concrete framework identity."""

from __future__ import annotations

from typing import Dict

from app.runtime.adapter import AgentRuntimeAdapter
from app.runtime.contracts import AgentDefinition
from app.runtime.mode import AgentRuntimeMode, agent_runtime_mode


def _default_langgraph_adapter() -> AgentRuntimeAdapter:
    """Select transport at process startup while retaining an emergency fallback."""

    if agent_runtime_mode() is AgentRuntimeMode.EXTERNAL:
        from app.runtime.http_adapter import HttpLangGraphRuntimeAdapter

        return HttpLangGraphRuntimeAdapter()
    from app.runtime.langgraph_adapter import LangGraphRuntimeAdapter

    return LangGraphRuntimeAdapter()


def _default_hermes_adapter() -> AgentRuntimeAdapter:
    from app.runtime.hermes_adapter import HermesRuntimeAdapter

    return HermesRuntimeAdapter()


class RuntimeSelectionError(ValueError):
    """Raised when a concrete definition cannot be routed to a runtime."""


class RuntimeRegistry:
    def __init__(self, adapters: list[AgentRuntimeAdapter] | None = None):
        self._adapters: Dict[tuple[str, str], AgentRuntimeAdapter] = {}
        self._initialized = adapters is not None
        for adapter in adapters or []:
            self.register(adapter)

    def _ensure_defaults(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self.register(_default_langgraph_adapter())
        self.register(_default_hermes_adapter())

    def initialize(self) -> None:
        """Resolve and import the configured adapters during application startup."""
        self._ensure_defaults()

    def register(self, adapter: AgentRuntimeAdapter) -> None:
        self._adapters[(adapter.framework, adapter.builder_id)] = adapter

    def get(self, definition: AgentDefinition) -> AgentRuntimeAdapter:
        self._ensure_defaults()
        key = (definition.framework, definition.builder_id)
        adapter = self._adapters.get(key)
        if adapter is None:
            raise RuntimeSelectionError(
                f"No runtime adapter for framework={definition.framework!r}, builder={definition.builder_id!r}"
            )
        if not definition.definition_id:
            raise RuntimeSelectionError("A concrete definition_id is required")
        return adapter


_registry = RuntimeRegistry()


def get_runtime_registry() -> RuntimeRegistry:
    return _registry


def adapter_for_definition(definition: AgentDefinition) -> AgentRuntimeAdapter:
    return get_runtime_registry().get(definition)
