"""CDP-backed viewport screenshot and input when Pelorus / Computer Use are unavailable."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Mapping

import requests
import websockets

logger = logging.getLogger(__name__)

BROWSER_DEBUG_URL = os.environ.get("BROWSER_DEBUG_URL", "localhost:9222")
_cdp_session = requests.Session()


def cdp_debug_reachable() -> bool:
    try:
        response = _cdp_session.get(f"http://{BROWSER_DEBUG_URL}/json", timeout=3)
        return response.status_code == 200 and bool(response.json())
    except Exception:
        return False


def _pick_active_tab() -> dict[str, Any]:
    response = _cdp_session.get(f"http://{BROWSER_DEBUG_URL}/json", timeout=5)
    response.raise_for_status()
    tabs = response.json()
    if not tabs:
        raise RuntimeError("No Brave tabs available")
    for tab in tabs:
        url = str(tab.get("url") or "")
        if url.startswith(("http://", "https://")):
            return tab
    for tab in tabs:
        url = str(tab.get("url") or "")
        if not url.startswith(("chrome://", "chrome-extension://", "about:")):
            return tab
    raise RuntimeError("No capturable browser tab")


async def _send_cdp_command(websocket, msg_id: int, method: str, params: dict[str, Any] | None = None):
    payload: dict[str, Any] = {"id": msg_id, "method": method}
    if params is not None:
        payload["params"] = params
    await websocket.send(json.dumps(payload))
    while True:
        response = await websocket.recv()
        result = json.loads(response)
        if result.get("id") == msg_id:
            return result


async def cdp_screenshot() -> dict[str, Any]:
    tab = _pick_active_tab()
    websocket_url = tab.get("webSocketDebuggerUrl")
    if not websocket_url:
        raise RuntimeError("Tab has no CDP websocket URL")
    async with websockets.connect(str(websocket_url), max_size=50 * 1024 * 1024) as websocket:
        await _send_cdp_command(websocket, 1, "Page.enable")
        result = await _send_cdp_command(
            websocket,
            2,
            "Page.captureScreenshot",
            {"format": "jpeg", "quality": 75, "fromSurface": True},
        )
        data = result.get("result", {}).get("data")
        if not data:
            raise RuntimeError("CDP captureScreenshot returned no image")
        return {
            "image": data,
            "screenshot": data,
            "desktop_control": True,
            "control_backend": "cdp",
            "url": tab.get("url"),
            "title": tab.get("title"),
        }


async def cdp_control(body: Mapping[str, Any]) -> dict[str, Any]:
    tab = _pick_active_tab()
    websocket_url = tab.get("webSocketDebuggerUrl")
    if not websocket_url:
        raise RuntimeError("Tab has no CDP websocket URL")
    action = str(body.get("action") or "")
    async with websockets.connect(str(websocket_url), max_size=10 * 1024 * 1024) as websocket:
        await _send_cdp_command(websocket, 1, "Input.enable", {})
        msg_id = 2
        if action in {"left_click", "click", "double_click", "mouse_move"}:
            coord = body.get("coordinate") or [0, 0]
            x, y = int(coord[0]), int(coord[1])
            await _send_cdp_command(
                websocket,
                msg_id,
                "Input.dispatchMouseEvent",
                {"type": "mouseMoved", "x": x, "y": y, "button": "left", "clickCount": 0},
            )
            msg_id += 1
            click_type = "mousePressed" if action == "mouse_move" else "mousePressed"
            if action != "mouse_move":
                await _send_cdp_command(
                    websocket,
                    msg_id,
                    "Input.dispatchMouseEvent",
                    {"type": click_type, "x": x, "y": y, "button": "left", "clickCount": 1},
                )
                msg_id += 1
                await _send_cdp_command(
                    websocket,
                    msg_id,
                    "Input.dispatchMouseEvent",
                    {"type": "mouseReleased", "x": x, "y": y, "button": "left", "clickCount": 1},
                )
        elif action == "type" and body.get("text"):
            await _send_cdp_command(websocket, msg_id, "Input.insertText", {"text": str(body["text"])})
        elif action == "scroll":
            direction = str(body.get("scroll_direction") or "down")
            amount = int(body.get("scroll_amount") or 3)
            delta_y = amount * 120 * (1 if direction == "down" else -1)
            await _send_cdp_command(
                websocket,
                msg_id,
                "Input.dispatchMouseEvent",
                {"type": "mouseWheel", "x": 0, "y": 0, "deltaX": 0, "deltaY": delta_y},
            )
        else:
            raise RuntimeError(f"Unsupported CDP control action: {action}")
    return {"status": "ok", "desktop_control": True, "control_backend": "cdp", "action": action}


async def cdp_state_summary() -> dict[str, Any]:
    tab = _pick_active_tab()
    return {
        "desktop_control": True,
        "control_backend": "cdp",
        "state": "cdp_viewport",
        "summary": f"Brave tab {tab.get('title') or tab.get('url')} (CDP viewport control; Selkies desktop stream is separate).",
        "url": tab.get("url"),
        "title": tab.get("title"),
    }