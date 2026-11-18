"""Shared parent → child delegation settings and child-run trace metadata."""

from __future__ import annotations

from typing import Any, Mapping

DELEGATE_INHERITED_TOOL_SETTINGS_KEYS = (
    "web_search_mode",
    "hitl_web_approval",
    "use_web_search",
    "browser_access_mode",
    "hitl_browser_approval",
    "use_browser_access",
    "hitl_policy",
    "hitl_canvas_publish",
)


def pick_delegate_inherited_tool_settings(source: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    picked: dict[str, Any] = {}
    for key in DELEGATE_INHERITED_TOOL_SETTINGS_KEYS:
        if key not in source:
            continue
        value = source[key]
        if value is None:
            continue
        picked[key] = value
    return _normalize_delegate_tool_derivatives(picked)


def merge_delegate_inherited_tool_settings(*sources: Mapping[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in sources:
        merged.update(pick_delegate_inherited_tool_settings(source))
    return _normalize_delegate_tool_derivatives(merged)


def _normalize_delegate_tool_derivatives(settings: dict[str, Any]) -> dict[str, Any]:
    if not settings:
        return settings
    mode = settings.get("web_search_mode")
    if mode in {"on", "off", "ask"}:
        settings.setdefault("use_web_search", mode != "off")
        settings.setdefault("hitl_web_approval", mode == "ask")
    browser_mode = settings.get("browser_access_mode")
    if browser_mode in {"on", "off", "ask"}:
        settings.setdefault("use_browser_access", browser_mode != "off")
        settings.setdefault("hitl_browser_approval", browser_mode == "ask")
    return settings


def append_delegated_child_run(
    prior: list[dict[str, Any]] | None,
    *,
    run_id: str,
    workflow_id: str,
) -> list[dict[str, Any]]:
    child_run_id = str(run_id or "").strip()
    if not child_run_id:
        return list(prior or [])
    child_workflow_id = str(workflow_id or "").strip()
    rows = [dict(row) for row in (prior or []) if isinstance(row, dict)]
    order = len(rows) + 1
    for row in rows:
        if str(row.get("run_id") or "").strip() == child_run_id:
            if child_workflow_id and not row.get("workflow_id"):
                row["workflow_id"] = child_workflow_id
            return rows
    rows.append({"run_id": child_run_id, "workflow_id": child_workflow_id, "order": order})
    return rows


def delegated_child_runs_from_technical(technical: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(technical, Mapping):
        return []
    rows: list[dict[str, Any]] = []
    raw = technical.get("delegated_child_runs")
    if isinstance(raw, list):
        for index, item in enumerate(raw, start=1):
            if not isinstance(item, Mapping):
                continue
            run_id = str(item.get("run_id") or "").strip()
            if not run_id:
                continue
            rows.append({
                "run_id": run_id,
                "workflow_id": str(item.get("workflow_id") or "").strip() or None,
                "order": int(item.get("order") or index),
            })
    if rows:
        return rows
    legacy_run_id = str(technical.get("child_run_id") or technical.get("delegate_child_run_id") or "").strip()
    if not legacy_run_id:
        return []
    legacy_workflow = str(technical.get("child_workflow_id") or technical.get("delegate_child_workflow_id") or "").strip()
    return [{
        "run_id": legacy_run_id,
        "workflow_id": legacy_workflow or None,
        "order": 1,
    }]


def delegated_child_runs_technical_details(
    result: Mapping[str, Any] | None,
    *,
    latest_run_id: str | None = None,
    latest_workflow_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        result = {}
    rows = append_delegated_child_run(
        delegated_child_runs_from_technical(result),
        run_id=str(latest_run_id or result.get("child_run_id") or result.get("delegate_child_run_id") or ""),
        workflow_id=str(latest_workflow_id or result.get("child_workflow_id") or result.get("delegate_child_workflow_id") or ""),
    )
    if not rows:
        return {}
    latest = rows[-1]
    details: dict[str, Any] = {
        "delegated_child_runs": rows,
        "child_run_id": latest.get("run_id"),
        "child_workflow_id": latest.get("workflow_id"),
        "delegate_child_run_id": latest.get("run_id"),
        "delegate_child_workflow_id": latest.get("workflow_id"),
    }
    agent_task_id = result.get("agent_task_id")
    if agent_task_id not in (None, ""):
        details["agent_task_id"] = agent_task_id
    return details