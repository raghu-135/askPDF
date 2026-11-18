from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.agent.tool_registry import (
    TOOL_FRIENDLY_CONFIG,
    apply_tool_instruction_key_aliases,
    build_tool_catalog_entry,
    format_model_tool_inventory_section,
    get_settings_tool_catalog_entries,
    get_workflow_prompt_tool_names,
    is_internal_tool,
)
from app.time_utils import iso_utc_z, parse_datetime_utc, utc_now


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


def _resolve_tool_items(
    tool_items: Optional[List[Any]],
    *,
    use_web_search: bool = True,
    use_browser_access: bool = False,
) -> List[str]:
    if tool_items is not None:
        return [_tool_name(item) for item in tool_items]
    return get_workflow_prompt_tool_names(
        use_web_search=use_web_search,
        use_browser_access=use_browser_access,
    )


def get_tool_catalog(
    tool_items: Optional[List[Any]] = None,
    *,
    use_web_search: bool = True,
    use_browser_access: bool = False,
    for_settings: bool = False,
) -> List[Dict[str, str]]:
    if for_settings and tool_items is None:
        return get_settings_tool_catalog_entries()
    catalog: List[Dict[str, str]] = []
    for tool_name in _resolve_tool_items(
        tool_items,
        use_web_search=use_web_search,
        use_browser_access=use_browser_access,
    ):
        if is_internal_tool(tool_name):
            continue
        catalog.append(build_tool_catalog_entry(tool_name))
    return catalog


def get_default_tool_instruction_map(tool_items: Optional[List[Any]] = None) -> Dict[str, str]:
    return {
        item["id"]: item["default_prompt"]
        for item in get_tool_catalog(tool_items, for_settings=tool_items is None)
    }


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
    aliased = apply_tool_instruction_key_aliases({str(k): str(v) for k, v in raw.items()})
    canonical_values = {
        tool_id: value
        for tool_id, value in aliased.items()
        if tool_id in normalized
    }
    known_ids = {item["id"] for item in TOOL_FRIENDLY_CONFIG.values()} | set(normalized)
    known_ids |= set(TOOL_FRIENDLY_CONFIG.keys())
    unknown = set(aliased) - known_ids
    if unknown:
        raise ValueError(f"Unknown tool instruction identifiers: {sorted(unknown)}")
    for tool_id, value in canonical_values.items():
        if tool_id not in normalized:
            continue
        text = _sanitize_lines_with_blocklist(str(value or ""), blocked, max_chars_per_tool)
        if text:
            normalized[tool_id] = text
    return normalized


def build_tool_registry_and_playbook_sections(
    tool_items: Optional[List[Any]],
    tool_instructions: Optional[Dict[str, str]],
    *,
    use_web_search: bool = True,
    use_browser_access: bool = False,
) -> Dict[str, str]:
    resolved_tools = _resolve_tool_items(
        tool_items,
        use_web_search=use_web_search,
        use_browser_access=use_browser_access,
    )
    catalog = get_tool_catalog(resolved_tools)
    playbook = normalize_tool_instructions(
        tool_instructions or {},
        tool_items=resolved_tools,
    )
    return {
        "TOOL_REGISTRY_SECTION": format_model_tool_inventory_section(catalog),
        "TOOL_PLAYBOOK_SECTION": "\n".join(
            [
                "## Tool Playbook",
                "",
                *(
                    f"- `{item['tool_name']}`: {playbook.get(item['id'], item['default_prompt'])}"
                    for item in catalog
                ),
            ]
        ),
    }


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