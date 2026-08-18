"""Framework-neutral external research handler."""

from app.agent.tool_contract import ToolWarningCode, make_tool_error_result, make_tool_result, tool_started
from app.tools.contracts import QueryRequest
from app.tools.context import ToolInvocationContext
from app.tools.provider_clients import PROVIDERS
from app.tools.provider_ports import ResearchProvider


async def search_external(request: QueryRequest, context: ToolInvocationContext, *, tool_name: str, provider: ResearchProvider | None = None):
    started = tool_started()
    try:
        provider = provider or PROVIDERS[tool_name]()
        result = await provider.search(request.query, context=context)
        warnings = [] if result.content.strip() else [ToolWarningCode.EMPTY_EXTERNAL_TOOL_RESULT]
        return make_tool_result(tool_name=tool_name, content=result.content, context=context, started=started, sources=result.sources, warnings=warnings, artifacts={"provider_tool": provider.__class__.__name__, "web_sources": result.sources if tool_name in {"yahoo_finance_news"} else []})
    except Exception as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=f"{tool_name} failed: {exc}")
