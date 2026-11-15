import importlib.util
import os
from pathlib import Path

import pytest

_REPO_ROOT = Path(os.environ.get("ASKPDF_REPO_DIR", Path(__file__).resolve().parents[2]))
_POLICY_PATH = _REPO_ROOT / "browser_capture" / "service" / "url_policy.py"
_spec = importlib.util.spec_from_file_location("browser_capture_url_policy", _POLICY_PATH)
assert _spec and _spec.loader
_url_policy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_url_policy)
BrowserUrlPolicyError = _url_policy.BrowserUrlPolicyError
validate_navigation_url = _url_policy.validate_navigation_url


def test_validate_navigation_url_accepts_https():
    assert validate_navigation_url("https://example.com/path") == "https://example.com/path"


def test_validate_navigation_url_rejects_about_blank():
    with pytest.raises(BrowserUrlPolicyError) as exc:
        validate_navigation_url("about:blank")
    assert exc.value.code == "browser_capture_blank_page"


def test_validate_navigation_url_rejects_localhost():
    with pytest.raises(BrowserUrlPolicyError) as exc:
        validate_navigation_url("http://localhost:3000/")
    assert exc.value.code == "browser_url_blocked"