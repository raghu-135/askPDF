"""Runtime adapter registry keyed by concrete framework identity."""

from __future__ import annotations

from typing import Dict

from app.runtime.adapter import AgentRuntimeAdapter
from app.runtime.contracts import AgentDefinition
from app.runtime.langgraph_adapter import LangGraphRuntimeAdapter


class RuntimeSelectionError(ValueError):
    """Raised when a concrete definition cannot be routed to a runtime."""


class RuntimeRegistry:
    def __init__(self, adapters: list[AgentRuntimeAdapter] | None = None):
        active = adapters or [LangGraphRuntimeAdapter()]
        self._adapters: Dict[tuple[str, str], AgentRuntimeAdapter] = {
            (adapter.framework, adapter.builder_id): adapter for adapter in active
        }

    def register(self, adapter: AgentRuntimeAdapter) -> None:
        self._adapters[(adapter.framework, adapter.builder_id)] = adapter

    def get(self, definition: AgentDefinition) -> AgentRuntimeAdapter:
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
