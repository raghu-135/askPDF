from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from app.api import files as files_api


def test_validate_browser_capture_url_rejects_about_blank():
    with pytest.raises(HTTPException) as exc:
        files_api._validate_browser_capture_url("about:blank")
    assert exc.value.status_code == 422
    assert exc.value.detail["code"] == "browser_capture_blank_page"


def test_raise_browser_capture_service_error_maps_upstream_422():
    request = httpx.Request("POST", "http://browser-capture:8080/capture")
    response = httpx.Response(
        422,
        request=request,
        json={
            "detail": {
                "code": "browser_capture_blank_page",
                "message": "Browser capture cannot index a blank or internal about: page. Open a web page first.",
            }
        },
    )
    error = httpx.HTTPStatusError("rejected", request=request, response=response)
    with pytest.raises(HTTPException) as exc:
        files_api._raise_browser_capture_service_error(error)
    assert exc.value.status_code == 422
    assert exc.value.detail["code"] == "browser_capture_blank_page"


def test_raise_browser_capture_service_error_hides_upstream_503_detail():
    request = httpx.Request("POST", "http://browser-capture:8080/capture")
    response = httpx.Response(503, request=request, text="upstream raw failure body")
    error = httpx.HTTPStatusError("unavailable", request=request, response=response)
    with pytest.raises(HTTPException) as exc:
        files_api._raise_browser_capture_service_error(error)
    assert exc.value.status_code == 503
    assert exc.value.detail["code"] == "browser_capture_unavailable"
    assert "upstream raw failure body" not in str(exc.value.detail)


@pytest.mark.asyncio
async def test_capture_browser_page_endpoint_maps_blank_capture_to_422():
    thread = MagicMock(id="thread-1")
    with (
        patch.object(files_api, "_require_ready_thread", new=AsyncMock(return_value=thread)),
        patch.object(
            files_api,
            "_capture_current_page",
            new=AsyncMock(
                side_effect=HTTPException(
                    status_code=422,
                    detail={
                        "code": "browser_capture_blank_page",
                        "message": "Browser capture cannot index a blank or internal about: page. Open a web page first.",
                    },
                )
            ),
        ),
    ):
        with pytest.raises(HTTPException) as exc:
            await files_api.capture_browser_page_endpoint("thread-1", MagicMock())
    assert exc.value.status_code == 422
