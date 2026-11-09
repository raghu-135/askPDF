"""Minimal Hermes gateway control-socket client for multiplex profile hot-serve."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONTROL_PROTOCOL_VERSION = 1
_SOCKET_FILENAME = "gateway.sock"
_POINTER_FILENAME = "gateway.sock.path"
_DEFAULT_TIMEOUT = 8.0


def gateway_home() -> Path:
    raw = os.getenv("HERMES_GATEWAY_HOME") or os.getenv("HERMES_DATA_ROOT") or "/opt/data"
    return Path(raw)


def _socket_path(home: Path) -> Path | None:
    direct = home / _SOCKET_FILENAME
    with suppress(OSError):
        if direct.exists():
            return direct
    with suppress(OSError):
        pointer = home / _POINTER_FILENAME
        if pointer.is_file():
            target = pointer.read_text(encoding="utf-8-sig").strip()
            if target and Path(target).exists():
                return Path(target)
    return None


def _query(home: Path, verb: str, *, params: dict[str, Any] | None = None, timeout: float = _DEFAULT_TIMEOUT) -> dict[str, Any] | None:
    path = _socket_path(home)
    if path is None:
        return None
    payload: dict[str, Any] = {"verb": verb, "id": 1, "protocol": _CONTROL_PROTOCOL_VERSION}
    if params:
        payload["params"] = params
    request = json.dumps(payload).encode("utf-8") + b"\n"
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(path))
            sock.sendall(request)
            chunks: list[bytes] = []
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                chunk = sock.recv(65536)
                chunks.append(chunk)
                if not chunk or b"\n" in chunk:
                    break
            raw = b"".join(chunks).partition(b"\n")[0]
            if not raw:
                return None
            response = json.loads(raw.decode("utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(response, dict) or response.get("ok") is not True:
        return None
    result = response.get("result")
    return result if isinstance(result, dict) else None


def ensure_profile_served(profile_name: str, *, timeout: float = _DEFAULT_TIMEOUT) -> bool:
    """Ask the default multiplexer to hot-serve a run profile directory that just appeared."""
    home = gateway_home()
    result = _query(home, "serve-profile", params={"name": profile_name}, timeout=timeout)
    if result is not None:
        logger.info("Hermes gateway hot-served profile %s", profile_name)
        return True
    result = _query(home, "rescan-profiles", timeout=timeout)
    if result is not None:
        served = {str(value) for value in result.get("served_profiles") or []}
        if profile_name in served:
            logger.info("Hermes gateway rescanned; profile %s is served", profile_name)
            return True
    logger.warning("Hermes gateway did not confirm hot-serve for profile %s", profile_name)
    return False


def release_profile_served(profile_name: str, *, timeout: float = _DEFAULT_TIMEOUT) -> None:
    """Best-effort unserve after a run profile is retired."""
    _query(gateway_home(), "unserve-profile", params={"name": profile_name}, timeout=timeout)
