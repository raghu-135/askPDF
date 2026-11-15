"""Pelorus proxy when available; else CDP viewport control on the shared Brave tab."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

import httpx

from cdp_desktop import cdp_control, cdp_debug_reachable, cdp_screenshot, cdp_state_summary

DESKTOP_CONTROL_UNAVAILABLE_MESSAGE = (
    "Desktop control is not available: Pelorus/Computer Use do not run on this X11+Brave CDP image "
    "and CDP debug is unreachable."
)

CDP_FALLBACK_NOTE = (
    "Pelorus and Selkies Computer Use (port 5000) are not enabled on this image; "
    "snapshot/act use CDP on the active Brave tab. The Browser tab stream is Selkies (X11), not Pelorus."
)


def pelorus_base_url() -> str:
    return (os.environ.get("PELORUS_INTERNAL_URL") or "http://127.0.0.1:3000/pelorus").rstrip("/")


def computer_use_base_url() -> str:
    return (os.environ.get("PIXELFLUX_CU_URL") or "http://127.0.0.1:5000").rstrip("/")


def pelorus_cu_token() -> str:
    return str(os.environ.get("PIXELFLUX_CU_TOKEN") or "").strip()


def pelorus_headers() -> dict[str, str]:
    token = pelorus_cu_token()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _pelorus_enabled() -> bool:
    return str(os.environ.get("PELORUS") or "").strip().lower() in {"1", "true", "yes"}


async def _pelorus_proxy_ready() -> bool:
    if not _pelorus_enabled():
        return False
    if not pelorus_cu_token():
        return False
    url = f"{pelorus_base_url()}/api/state"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=pelorus_headers(), timeout=5.0)
            return response.status_code == 200
    except Exception:
        return False


async def _computer_use_ready() -> bool:
    if not pelorus_cu_token():
        return False
    url = f"{computer_use_base_url()}/computer-use"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json={"action": "cursor_position"},
                headers=pelorus_headers(),
                timeout=2.0,
            )
            return response.status_code == 200
    except Exception:
        return False


async def pelorus_ready() -> bool:
    if await _pelorus_proxy_ready():
        return True
    if await _computer_use_ready():
        return True
    return cdp_debug_reachable()


async def pelorus_get(path: str, *, timeout: float = 30.0) -> dict[str, Any]:
    if await _pelorus_proxy_ready():
        url = f"{pelorus_base_url()}{path if path.startswith('/') else '/' + path}"
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=pelorus_headers(), timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {"data": payload}
    if cdp_debug_reachable():
        summary = await cdp_state_summary()
        summary["note"] = CDP_FALLBACK_NOTE
        return summary
    return {
        "desktop_control": False,
        "code": "desktop_control_unavailable",
        "message": DESKTOP_CONTROL_UNAVAILABLE_MESSAGE,
        "summary": CDP_FALLBACK_NOTE,
        "state": "unavailable",
    }


async def _computer_use_post(body: Mapping[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
    url = f"{computer_use_base_url()}/computer-use"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=dict(body), headers=pelorus_headers(), timeout=timeout)
        response.raise_for_status()
        text = response.text.strip()
        if not text:
            return {"status": "ok"}
        try:
            payload = response.json()
        except json.JSONDecodeError:
            payload = json.loads(text) if text.startswith("{") else {"data": text}
        return payload if isinstance(payload, dict) else {"data": payload}


def _map_control_action(action: str) -> str:
    if action == "click":
        return "left_click"
    return action


async def pelorus_post(path: str, body: Mapping[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
    if await _pelorus_proxy_ready():
        url = f"{pelorus_base_url()}{path if path.startswith('/') else '/' + path}"
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=dict(body), headers=pelorus_headers(), timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {"data": payload}

    payload = dict(body)
    payload["action"] = _map_control_action(str(payload.get("action") or ""))
    if await _computer_use_ready():
        if path.rstrip("/").endswith("screenshot") or payload.get("action") == "screenshot":
            return await _computer_use_post({"action": "screenshot"}, timeout=timeout)
        return await _computer_use_post(payload, timeout=timeout)

    if cdp_debug_reachable():
        if path.rstrip("/").endswith("screenshot") or payload.get("action") == "screenshot":
            return await cdp_screenshot()
        if payload.get("action") != "screenshot":
            return await cdp_control(payload)

    raise RuntimeError(DESKTOP_CONTROL_UNAVAILABLE_MESSAGE)


async def desktop_screenshot(*, timeout: float = 45.0) -> dict[str, Any]:
    del timeout
    if await _pelorus_proxy_ready():
        return await pelorus_get("/api/desktop/screenshot")
    if await _computer_use_ready():
        return await _computer_use_post({"action": "screenshot"})
    return await cdp_screenshot()