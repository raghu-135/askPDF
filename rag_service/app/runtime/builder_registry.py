"""Registry for concrete framework-specific builder providers."""

from __future__ import annotations

from typing import Dict

from app.runtime.builder import AgentBuilderProvider
from runtime_protocol.contracts import AgentDefinition


class BuilderSelectionError(ValueError):
    """Raised when no provider is registered for a concrete identity."""


class BuilderRegistry:
    def __init__(self, providers: list[AgentBuilderProvider] | None = None):
        self._providers: Dict[tuple[str, str], AgentBuilderProvider] = {}
        for provider in providers or [self._default_langgraph_provider(), self._default_hermes_provider()]:
            self.register(provider)

    @staticmethod
    def _default_langgraph_provider() -> AgentBuilderProvider:
        from app.runtime.langgraph_builder import LangGraphBuilderProvider

        return LangGraphBuilderProvider()

    @staticmethod
    def _default_hermes_provider() -> AgentBuilderProvider:
        from app.runtime.hermes_builder import HermesBuilderProvider

        return HermesBuilderProvider()

    def register(self, provider: AgentBuilderProvider) -> None:
        self._providers[(provider.framework, provider.builder_id)] = provider

    def get(self, definition: AgentDefinition) -> AgentBuilderProvider:
        if not definition.definition_id:
            raise BuilderSelectionError("A concrete definition_id is required")
        key = (definition.framework, definition.builder_id)
        provider = self._providers.get(key)
        if provider is None:
            raise BuilderSelectionError(
                f"No builder provider for framework={definition.framework!r}, builder={definition.builder_id!r}"
            )
        return provider

    def get_by_identity(self, framework: str, builder_id: str) -> AgentBuilderProvider:
        provider = self._providers.get((framework, builder_id))
        if provider is None:
            raise BuilderSelectionError(
                f"No builder provider for framework={framework!r}, builder={builder_id!r}"
            )
        return provider

    def providers(self) -> tuple[AgentBuilderProvider, ...]:
        return tuple(self._providers.values())


_registry = BuilderRegistry()


def get_builder_registry() -> BuilderRegistry:
    return _registry


def builder_for_definition(definition: AgentDefinition) -> AgentBuilderProvider:
    return get_builder_registry().get(definition)
