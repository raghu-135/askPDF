"""Derive authoritative web/browser permission modes from legacy knobs."""

from __future__ import annotations

from typing import Any, Mapping


def derive_web_search_mode(
    *,
    web_search_mode: Any = None,
    hitl_web_approval: Any = None,
    use_web_search: Any = None,
    config: Mapping[str, Any] | None = None,
) -> str:
    """Prefer explicit web_search_mode; otherwise derive ask/on/off."""

    config = config or {}
    for source in (web_search_mode, config.get("web_search_mode")):
        if source in {"on", "off", "ask"}:
            return str(source)
    hitl = hitl_web_approval if hitl_web_approval is not None else config.get("hitl_web_approval")
    if hitl:
        return "ask"
    use = use_web_search if use_web_search is not None else config.get("use_web_search")
    if use:
        return "on"
    return "off"


def derive_browser_access_mode(
    *,
    browser_access_mode: Any = None,
    hitl_browser_approval: Any = None,
    use_browser_access: Any = None,
    config: Mapping[str, Any] | None = None,
) -> str:
    """Prefer explicit browser_access_mode; otherwise derive ask/on/off."""

    config = config or {}
    for source in (browser_access_mode, config.get("browser_access_mode")):
        if source in {"on", "off", "ask"}:
            return str(source)
    hitl = hitl_browser_approval if hitl_browser_approval is not None else config.get("hitl_browser_approval")
    if hitl:
        return "ask"
    use = use_browser_access if use_browser_access is not None else config.get("use_browser_access")
    if use:
        return "on"
    return "off"