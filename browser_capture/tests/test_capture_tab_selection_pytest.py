from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from service.main import NoCapturableBrowserTab, app


@pytest.fixture(name="client")
def client_fixture():
    return TestClient(app)


def test_capture_returns_422_when_only_about_blank_tab(client):
    with patch(
        "service.main.get_active_tab_via_cdp",
        side_effect=NoCapturableBrowserTab("No capturable http(s) tab."),
    ):
        response = client.post("/capture")
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "browser_capture_no_http_tab"
    assert "http(s)" in detail["message"]
