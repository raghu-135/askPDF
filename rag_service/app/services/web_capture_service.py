import os
import hashlib
import logging
import json
import asyncio
import base64
from datetime import datetime
from typing import Optional

import nodriver as uc
from nodriver import cdp
from markitdown import MarkItDown

logger = logging.getLogger(__name__)

# Base directories for shared web captures and PDFs
WEBPAGES_DIR = "/static/webpages"
STATIC_DIR = "/static"
os.makedirs(WEBPAGES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Common tracker/cookie banner domains to block
BLOCKED_DOMAINS = {
    "google-analytics.com",
    "googletagmanager.com",
    "googleadservices.com",
    "doubleclick.net",
    "facebook.com",
    "fbcdn.net",
    "twitter.com",
    "analytics.twitter.com",
    "linkedin.com",
    "licdn.com",
    "hotjar.com",
    "optimizely.com",
    "segment.com",
    "mixpanel.com",
    "amplitude.com",
    "intercom.io",
    "driftt.com",
    "zendesk.com",
    "zdassets.com",
    "cookiebot.com",
    "onetrust.com",
    "trustarc.com",
    "quantserve.com",
    "scorecardresearch.com",
    "moatads.com",
    "outbrain.com",
    "taboola.com",
    "revcontent.com",
    "adsystem.amazon.com",
    "amazon-adsystem.com",
}


def _url_to_hash(url: str) -> str:
    """Generate a stable MD5 hash of a URL for file naming."""
    return hashlib.md5(url.encode()).hexdigest()


def _should_block_url(url: str) -> bool:
    """Check if URL should be blocked (trackers, ads, etc)."""
    url_lower = url.lower()
    for blocked in BLOCKED_DOMAINS:
        if blocked in url_lower:
            return True
    return False


async def _dismiss_cookie_banner(tab) -> bool:
    """
    Attempt to find and click cookie consent buttons.
    Returns True if a button was clicked.
    """
    cookie_texts = [
        "accept all", "agree", "accept", "allow all", "ok",
        "i agree", "consent", "accept cookies", "allow cookies",
        "got it", "i understand", "agree to all", "enable all",
        "yes, accept", "accept all cookies", "allow all cookies"
    ]

    for text in cookie_texts:
        try:
            button = await tab.find(text, timeout=1.5)
            if button:
                await button.click()
                await tab.sleep(0.5)
                logger.info(f"Clicked cookie banner button: '{text}'")
                return True
        except Exception:
            continue

    return False


async def _scroll_to_lazy_load(tab) -> None:
    """
    Scroll the page gradually to trigger lazy-loaded images and content.
    """
    # Get page height
    try:
        page_height = await tab.evaluate("document.body.scrollHeight")
        viewport_height = await tab.evaluate("window.innerHeight")

        if page_height and viewport_height:
            scroll_steps = max(5, min(15, page_height // viewport_height))
            for i in range(scroll_steps):
                await tab.evaluate(f"window.scrollBy(0, {viewport_height * 0.8})")
                await tab.sleep(0.3)
            # Scroll back to top
            await tab.evaluate("window.scrollTo(0, 0)")
            await tab.sleep(0.3)
    except Exception as e:
        logger.warning(f"Error during lazy-load scrolling: {e}")
        # Fallback simple scroll
        for _ in range(5):
            await tab.evaluate("window.scrollBy(0, 500)")
            await tab.sleep(0.3)


async def _render_page_with_nodriver(url: str) -> tuple[str, bytes, str]:
    """
    Render a webpage using nodriver with Chrome.
    Handles cookie banners, lazy-loaded images, and generates PDF.

    Args:
        url: The URL to render

    Returns:
        tuple of (html_content: str, pdf_bytes: bytes, title: str)
    """
    browser = None
    try:
        # Start browser with anti-detection settings
        browser = await uc.start(
            headless=True,
            browser_executable_path="/usr/bin/chromium",
            browser_args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
            ]
        )

        tab = await browser.get(url)

        # Wait for page to load
        await tab.sleep(2)

        # Try to dismiss cookie banner
        await _dismiss_cookie_banner(tab)

        # Scroll to trigger lazy-loaded content
        await _scroll_to_lazy_load(tab)

        # Wait a bit more for any triggered content to load
        await tab.sleep(1.5)

        # Get page title
        try:
            title = await tab.evaluate("document.title")
            if not title or title.strip() in ("", "about:blank"):
                title = url
        except Exception:
            title = url

        # Inject CSS for continuous/scrollable PDF (no page breaks)
        print_css = """
        @page {
            margin: 0 !important;
            size: auto !important;
        }

        * {
            page-break-inside: auto !important;
            break-inside: auto !important;
            page-break-before: auto !important;
            break-before: auto !important;
            page-break-after: auto !important;
            break-after: auto !important;
        }

        @media print {
            body {
                margin: 0 !important;
                padding: 0 !important;
                -webkit-print-color-adjust: exact !important;
                print-color-adjust: exact !important;
            }
        }
        """
        try:
            await tab.evaluate(f"""
                const style = document.createElement('style');
                style.textContent = `{print_css}`;
                document.head.appendChild(style);
            """)
            await tab.sleep(0.5)
        except Exception as e:
            logger.warning(f"Failed to inject print CSS: {e}")

        # Get HTML content
        try:
            html_content = await tab.evaluate("document.documentElement.outerHTML")
        except Exception as e:
            logger.warning(f"Failed to get HTML content: {e}")
            html_content = f"<html><body><p>Failed to retrieve content from {url}</p></body></html>"

        # Generate PDF using Chrome's CDP via tab connection
        pdf_bytes = b""
        try:
            # Use tab's connection to send CDP command
            pdf_response = await tab.send(
                cdp.page.print_to_pdf(
                    landscape=False,
                    print_background=True,
                    margin_top=0,
                    margin_bottom=0,
                    margin_left=0,
                    margin_right=0,
                    paper_width=8.27,  # A4 width in inches
                    paper_height=11.69,  # A4 height in inches
                    prefer_css_page_size=False,  # Use explicit paper size
                )
            )
            # CDP returns a response object with 'data' attribute
            if pdf_response:
                if hasattr(pdf_response, 'data'):
                    pdf_bytes = base64.b64decode(pdf_response.data)
                elif isinstance(pdf_response, (list, tuple)) and len(pdf_response) > 0:
                    # Sometimes CDP returns a tuple/list
                    pdf_bytes = base64.b64decode(pdf_response[0])
                else:
                    logger.warning(f"Unexpected PDF response: {type(pdf_response)} - {pdf_response}")
        except Exception as e:
            logger.error(f"Failed to generate PDF with CDP: {e}", exc_info=True)

        if not pdf_bytes:
            logger.error("PDF generation failed - empty PDF bytes")
            raise RuntimeError("Failed to generate PDF from webpage")

        return html_content, pdf_bytes, title

    except Exception as e:
        logger.error(f"Error rendering page with nodriver: {e}", exc_info=True)
        raise RuntimeError(f"Failed to capture webpage: {str(e)}")

    finally:
        if browser:
            try:
                browser.stop()
            except Exception:
                pass


def _extract_clean_markdown(html_content: str, url: str) -> str:
    """
    Extract clean markdown from HTML using MarkItDown.

    Args:
        html_content: Raw HTML string
        url: Source URL for context

    Returns:
        Clean markdown string
    """
    try:
        md = MarkItDown()
        result = md.convert_string(html_content)
        if result and result.text_content.strip():
            logger.info(f"MarkItDown successfully extracted markdown from {url}")
            return result.text_content
    except Exception as e:
        logger.warning(f"MarkItDown extraction failed: {e}")

    # Fallback if extraction fails
    return f"# {url}\n\nContent could not be extracted from this page."


async def capture_webpage(url: str, force: bool = False) -> dict:
    """
    Capture a webpage using nodriver (Chrome) and extract clean markdown.

    Pipeline:
    1. nodriver (Chrome) renders the page with JS, handles cookie banners
    2. Extract HTML snapshot + PDF (for display)
    3. HTML → MarkItDown → clean Markdown (for vector DB)
    4. Save PDF to /static/{hash}.pdf
    5. Save mapping file for URL→PDF lookup

    Args:
        url: The URL to capture
        force: If True, force recapture even if already cached

    Returns:
        dict with file_hash (PDF hash), url_hash, title, pdf_path, original_url,
        source_type, markdown_content, and cached flag
    """
    url_hash = _url_to_hash(url)
    mapping_path = os.path.join(WEBPAGES_DIR, f"{url_hash}.mapping.json")

    # Check if we have a cached mapping and PDF
    if not force and os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            pdf_path = mapping.get("pdf_path")
            if pdf_path and os.path.exists(pdf_path):
                logger.info(f"Using cached capture for {url}")
                return {
                    "file_hash": mapping["pdf_hash"],
                    "url_hash": url_hash,
                    "title": mapping["title"],
                    "pdf_path": pdf_path,
                    "original_url": url,
                    "source_type": "web",
                    "markdown_content": mapping.get("markdown_content", ""),
                    "cached": True,
                }
        except Exception as e:
            logger.warning(f"Failed to read cached mapping for {url}: {e}")

    # Capture webpage
    logger.info(f"Capturing webpage: {url}")

    try:
        # Step 1: Render with nodriver (Chrome)
        html_content, pdf_bytes, title = await _render_page_with_nodriver(url)

        # Step 2: Extract clean markdown
        markdown_content = _extract_clean_markdown(html_content, url)

        # Step 3: Calculate PDF hash and save
        pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
        pdf_path = os.path.join(STATIC_DIR, f"{pdf_hash}.pdf")

        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Step 4: Save mapping file
        mapping = {
            "url": url,
            "pdf_hash": pdf_hash,
            "pdf_path": pdf_path,
            "title": title,
            "markdown_content": markdown_content,
            "captured_at": datetime.utcnow().isoformat(),
        }
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

        logger.info(f"Captured {url} to {pdf_path} (hash: {pdf_hash})")

        return {
            "file_hash": pdf_hash,
            "url_hash": url_hash,
            "title": title,
            "pdf_path": pdf_path,
            "original_url": url,
            "source_type": "web",
            "markdown_content": markdown_content,
            "cached": False,
        }

    except Exception as e:
        logger.error(f"Failed to capture {url}: {e}", exc_info=True)
        raise RuntimeError(f"Failed to capture webpage: {str(e)}")


# Alias for backward compatibility during transition
async def capture_webpage_as_pdf(url: str, force: bool = False) -> dict:
    """Backward-compatible alias for capture_webpage."""
    return await capture_webpage(url, force)


async def get_webpage_pdf_by_url_hash(url_hash: str) -> Optional[dict]:
    """
    Look up a webpage's PDF by its URL hash.
    Used for refresh operations to find the current PDF before recapturing.

    Args:
        url_hash: The MD5 hash of the URL

    Returns:
        dict with pdf_hash, pdf_path, url, title, markdown_content or None if not found
    """
    mapping_path = os.path.join(WEBPAGES_DIR, f"{url_hash}.mapping.json")
    if not os.path.exists(mapping_path):
        return None

    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)

        # Verify PDF still exists
        if not os.path.exists(mapping.get("pdf_path", "")):
            return None

        return {
            "pdf_hash": mapping["pdf_hash"],
            "pdf_path": mapping["pdf_path"],
            "url": mapping["url"],
            "title": mapping["title"],
            "markdown_content": mapping.get("markdown_content", ""),
        }
    except Exception as e:
        logger.warning(f"Failed to read mapping for {url_hash}: {e}")
        return None
