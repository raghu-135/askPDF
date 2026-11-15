"""Thin HTTP client for Pelorus on the shared Selkies desktop."""

from __future__ import annotations

import os
from typing import Any, Mapping

import httpx


def pelorus_base_url() -> str:
    return (os.environ.get("PELORUS_INTERNAL_URL") or "http://127.0.0.1:3000/pelorus").rstrip("/")


def pelorus_cu_token() -> str:
    return str(os.environ.get("PIXELFLUX_CU_TOKEN") or "").strip()


def pelorus_headers() -> dict[str, str]:
    token = pelorus_cu_token()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


async def pelorus_get(path: str, *, timeout: float = 30.0) -> dict[str, Any]:
    url = f"{pelorus_base_url()}{path if path.startswith('/') else '/' + path}"
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=pelorus_headers(), timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"data": payload}


async def pelorus_post(path: str, body: Mapping[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
    url = f"{pelorus_base_url()}{path if path.startswith('/') else '/' + path}"
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=dict(body), headers=pelorus_headers(), timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else {"data": payload}


async def pelorus_ready() -> bool:
    if not pelorus_cu_token():
        return False
    try:
        await pelorus_get("/api/state", timeout=5.0)
        return True
    except Exception:
        return False