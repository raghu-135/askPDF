"""HTTP client for the shared browser-capture container."""

from __future__ import annotations

import os
from typing import Any, Mapping

import httpx

CAPTURE_SERVICE_URL = os.environ.get("CAPTURE_SERVICE_URL", "http://browser-capture:8080").rstrip("/")


class BrowserCaptureClientError(RuntimeError):
    def __init__(self, code: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _extract_detail(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {"message": response.text or response.reason_phrase}
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, dict):
            return detail
        if detail is not None:
            return {"message": str(detail)}
        return payload
    return {"message": str(payload)}


async def capture_navigate(url: str, *, timeout: float = 90.0) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{CAPTURE_SERVICE_URL}/navigate",
            json={"url": url},
            timeout=timeout,
        )
        if response.is_error:
            detail = _extract_detail(response)
            raise BrowserCaptureClientError(
                str(detail.get("code") or "browser_navigate_failed"),
                str(detail.get("message") or detail),
                status_code=response.status_code,
            )
        return response.json()


async def capture_read_page(*, timeout: float = 60.0) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{CAPTURE_SERVICE_URL}/read", timeout=timeout)
        if response.is_error:
            detail = _extract_detail(response)
            raise BrowserCaptureClientError(
                str(detail.get("code") or "browser_read_failed"),
                str(detail.get("message") or detail),
                status_code=response.status_code,
            )
        return response.json()


async def capture_current_page(*, timeout: float = 120.0) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{CAPTURE_SERVICE_URL}/capture", timeout=timeout)
        if response.is_error:
            detail = _extract_detail(response)
            raise BrowserCaptureClientError(
                str(detail.get("code") or "browser_capture_failed"),
                str(detail.get("message") or detail),
                status_code=response.status_code,
            )
        capture = response.json()
    capture["pdf_path"] = f"/captures/{capture['file_hash']}.pdf"
    return capture


async def pelorus_state(*, timeout: float = 30.0) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{CAPTURE_SERVICE_URL}/pelorus/state", timeout=timeout)
        if response.is_error:
            detail = _extract_detail(response)
            raise BrowserCaptureClientError(
                str(detail.get("code") or "pelorus_unavailable"),
                str(detail.get("message") or detail),
                status_code=response.status_code,
            )
        return response.json()


async def pelorus_screenshot(*, timeout: float = 45.0) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{CAPTURE_SERVICE_URL}/pelorus/screenshot", timeout=timeout)
        if response.is_error:
            detail = _extract_detail(response)
            raise BrowserCaptureClientError(
                str(detail.get("code") or "pelorus_unavailable"),
                str(detail.get("message") or detail),
                status_code=response.status_code,
            )
        return response.json()


async def pelorus_control(body: Mapping[str, Any], *, timeout: float = 60.0) -> dict[str, Any]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{CAPTURE_SERVICE_URL}/pelorus/control",
            json=dict(body),
            timeout=timeout,
        )
        if response.is_error:
            detail = _extract_detail(response)
            raise BrowserCaptureClientError(
                str(detail.get("code") or "pelorus_control_failed"),
                str(detail.get("message") or detail),
                status_code=response.status_code,
            )
        return response.json()