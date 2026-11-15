import os
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(os.environ.get("ASKPDF_REPO_DIR", Path(__file__).resolve().parents[2]))
_SERVICE_DIR = _REPO_ROOT / "browser_capture" / "service"
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))

from url_policy import BrowserUrlPolicyError, validate_navigation_url


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