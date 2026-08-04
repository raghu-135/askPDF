from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.agent.external_research_tools import get_external_research_tools
from app.agent.tool_registry import TOOL_FRIENDLY_CONFIG
from app.time_utils import iso_utc_z, parse_datetime_utc, utc_now


CORE_TOOL_NAMES = [
    "get_thread_shape",
    "search_documents",
    "search_document_by_id",
    "search_thread_conversation_history",
    "search_durable_memory",
    "search_thread_events",
    "search_web",
    "ask_for_clarification",
]

LEGACY_TOOL_INSTRUCTION_IDS = {
    "deep_memory": "thread_conversation_history",
    "memory_recall": "durable_memory",
    "thread_timeline": "thread_events",
}


def format_runtime_datetime_context(
    client_timezone: Optional[str] = None,
    client_locale: Optional[str] = None,
    client_now_iso: Optional[str] = None,
    now_utc: Optional[datetime] = None,
) -> str:
    """
    Build a locked runtime clock block for prompts.

    The browser supplies user-local timezone/locale; the server clock remains
    authoritative so a misconfigured client clock cannot redefine now.
    """
    server_now_utc = parse_datetime_utc(now_utc) or utc_now()

    timezone_name = (client_timezone or "").strip()[:100] or "UTC"
    timezone_note = ""
    try:
        user_tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        timezone_note = f"Browser timezone '{timezone_name}' was not recognized; UTC is used."
        timezone_name = "UTC"
        user_tz = timezone.utc

    user_now = server_now_utc.astimezone(user_tz)
    locale = (client_locale or "").strip()[:50] or "unknown"
    client_now = parse_datetime_utc(client_now_iso)
    skew_note = ""
    if client_now:
        skew_seconds = abs((server_now_utc - client_now).total_seconds())
        if skew_seconds > 300:
            skew_note = (
                f"Browser clock differs from server UTC by about {round(skew_seconds / 60)} minutes; "
                "server time is authoritative."
            )

    lines = [
        "## RUNTIME DATE/TIME CONTEXT (LOCKED - not overridable)",
        "",
        f"User-local current datetime: {user_now.isoformat(timespec='seconds')}",
        f"User timezone: {timezone_name}",
        f"User locale: {locale}",
        f"Server current UTC datetime: {iso_utc_z(server_now_utc).split('.')[0]}Z",
    ]
    if client_now_iso:
        lines.append(f"Browser-reported UTC datetime: {client_now_iso.strip()[:80]}")
    if timezone_note:
        lines.append(f"Timezone note: {timezone_note}")
    if skew_note:
        lines.append(f"Clock note: {skew_note}")
    lines.extend(
        [
            "",
            "Use this context to interpret relative date phrases such as today, yesterday, tomorrow, this week, last month, latest, and current.",
            "This clock does not make your knowledge current; for facts that may have changed recently, use retrieval or web search when available.",
        ]
    )
    return "\n".join(lines)


def _tool_name(tool_item: Any) -> str:
    return tool_item if isinstance(tool_item, str) else str(getattr(tool_item, "name", tool_item))


def _tool_description(tool_item: Any, tool_name: str) -> str:
    if not isinstance(tool_item, str):
        return str(getattr(tool_item, "description", "") or "")
    return ""


def _default_tool_names(use_external_research: bool = True) -> List[Any]:
    if not use_external_research:
        return list(CORE_TOOL_NAMES)
    external_names = [getattr(tool, "name", "") for tool in get_external_research_tools()]
    return [*CORE_TOOL_NAMES, *[name for name in external_names if name]]


def _sanitize_lines_with_blocklist(raw: str, blocklist: List[str], max_chars: int) -> str:
    if not raw:
        return ""
    lines = []
    for line in raw.splitlines():
        check = line.strip().lower()
        if any(bad in check for bad in blocklist):
            continue
        lines.append(line)
    return "\n".join(lines).strip()[:max_chars]


def sanitize_system_role(raw: str, max_chars: int = 500) -> str:
    blocked = [
        "ignore previous instructions",
        "you have no restrictions",
    ]
    return _sanitize_lines_with_blocklist(raw, blocked, max_chars)


def sanitize_custom_instructions(raw: str, max_chars: int = 2000) -> str:
    blocked = [
        "ignore previous instructions",
        "ignore all previous",
        "do not use tools",
        "disable tools",
        "never use tools",
        "pretend you have no tool",
    ]
    return _sanitize_lines_with_blocklist(raw, blocked, max_chars)


def get_tool_catalog(tool_items: Optional[List[Any]] = None) -> List[Dict[str, str]]:
    catalog: List[Dict[str, str]] = []
    for tool_item in tool_items or _default_tool_names(use_external_research=True):
        tool_name = _tool_name(tool_item)
        cfg = TOOL_FRIENDLY_CONFIG.get(tool_name, {})
        alias_id = str(cfg.get("id", tool_name))
        catalog.append(
            {
                "tool_name": tool_name,
                "id": alias_id,
                "display_name": str(cfg.get("display_name", alias_id.replace("_", " ").title())),
                "description": str(cfg.get("description", _tool_description(tool_item, tool_name))),
                "default_prompt": str(cfg.get("default_prompt", "Use this tool when it is the most relevant retrieval path.")),
            }
        )
    return catalog


def get_default_tool_instruction_map(tool_items: Optional[List[Any]] = None) -> Dict[str, str]:
    return {item["id"]: item["default_prompt"] for item in get_tool_catalog(tool_items)}


def normalize_tool_instructions(
    raw: Optional[Dict[str, str]],
    max_chars_per_tool: int = 500,
    tool_items: Optional[List[Any]] = None,
) -> Dict[str, str]:
    blocked = [
        "do not use tools",
        "disable tools",
        "never use tools",
        "ignore tool contract",
    ]
    normalized = get_default_tool_instruction_map(tool_items)
    if not isinstance(raw, dict):
        return normalized
    canonical_values = {
        tool_id: value
        for tool_id, value in raw.items()
        if tool_id in normalized
    }
    legacy_values = {
        LEGACY_TOOL_INSTRUCTION_IDS[tool_id]: value
        for tool_id, value in raw.items()
        if tool_id in LEGACY_TOOL_INSTRUCTION_IDS
        and LEGACY_TOOL_INSTRUCTION_IDS[tool_id] not in canonical_values
    }
    for tool_id, value in {**legacy_values, **canonical_values}.items():
        if tool_id not in normalized:
            continue
        text = _sanitize_lines_with_blocklist(str(value or ""), blocked, max_chars_per_tool)
        if text:
            normalized[tool_id] = text
    return normalized
