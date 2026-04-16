"""Web Capture module - PDF generation from URLs with consent handling."""

from app.web_capture.cmp_registry import CMPType, detect_cmp, get_strategy_for_cmp
from app.web_capture.capture_strategy import (
    CaptureOptions,
    ConsentStrategy,
    RenderStrategy,
    SuppressionValidator,
)
from app.web_capture.cookie_presets import Cookie, get_minimal_cookies_for_cmp
from app.web_capture.web_capture_service import (
    capture_webpage,
    capture_webpage_as_pdf,
    get_webpage_pdf_by_url_hash,
)

__all__ = [
    # CMP Registry
    "CMPType",
    "detect_cmp",
    "get_strategy_for_cmp",
    # Capture Strategy
    "CaptureOptions",
    "ConsentStrategy",
    "RenderStrategy",
    "SuppressionValidator",
    # Cookie Presets
    "Cookie",
    "get_minimal_cookies_for_cmp",
    # Web Capture Service
    "capture_webpage",
    "capture_webpage_as_pdf",
    "get_webpage_pdf_by_url_hash",
]
