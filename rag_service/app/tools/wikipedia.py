"""Framework-neutral Wikipedia request contract."""

from pydantic import BaseModel, Field

from app.tools.context import ToolInvocationContext
from app.agent.tool_contract import ToolResult
from app.tools.external_research import search_external
from app.tools.provider_clients import WikipediaProvider


class WikipediaRequest(BaseModel):
    query: str = Field(description="Short entity or topic to look up on Wikipedia.")


async def invoke_wikipedia(request: WikipediaRequest, context: ToolInvocationContext) -> ToolResult:
    return await search_external(request, context, tool_name="wikipedia", provider=WikipediaProvider())
