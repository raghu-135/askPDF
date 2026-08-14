"""Provider ports used by framework-neutral external research handlers."""

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.tools.context import ToolInvocationContext


@dataclass(frozen=True)
class ProviderResult:
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ResearchProvider(Protocol):
    name: str

    async def search(self, query: str, *, context: ToolInvocationContext) -> ProviderResult:
        ...
