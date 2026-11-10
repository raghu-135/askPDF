"""Prove askPDF's pinned Hermes patch is active in the running gateway process."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

def _gateway_proof_enabled() -> bool:
    flags = (
        os.getenv("HERMES_GATEWAY_PATCH_PROOF", ""),
        os.getenv("HERMES_RUNTIME_INTEGRATION", ""),
    )
    return any(value.lower() in {"1", "true", "yes", "on"} for value in flags)


pytestmark = pytest.mark.skipif(
    not _gateway_proof_enabled(),
    reason="requires HERMES_GATEWAY_PATCH_PROOF or HERMES_RUNTIME_INTEGRATION",
)


def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        pytest.skip(f"{name} is required for live Hermes gateway patch proof")
    return value


def _fetch_toolsets() -> dict:
    api_url = _require("HERMES_API_URL").rstrip("/")
    token = _require("HERMES_API_TOKEN")
    request = urllib.request.Request(
        f"{api_url}/p/askpdf-deep-offline/v1/toolsets",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise AssertionError(f"toolsets request failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise AssertionError(f"toolsets request failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise AssertionError("toolsets response must be a JSON object")
    return payload


def test_live_gateway_toolsets_include_askpdf_runtime_profile_metadata():
    """Stock Hermes does not add askpdf_runtime_profile; only our sitecustomize hook does."""
    payload = _fetch_toolsets()
    profile = payload.get("askpdf_runtime_profile")
    assert isinstance(profile, dict), payload
    assert "registered_tools" in profile
    assert "mcp_server_names" in profile
    assert "mcp_context_header_sha256" in profile

