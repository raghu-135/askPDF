import os
import hashlib
import logging
import json
import asyncio
from datetime import datetime
from typing import Optional

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
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


async def _render_page_with_playwright(url: str) -> tuple[str, bytes, str]:
    """
    Render a webpage using Playwright with Chromium.
    Blocks trackers, cookie banners, and other junk.

    Args:
        url: The URL to render

    Returns:
        tuple of (html_content: str, pdf_bytes: bytes, title: str)
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # Block unwanted resources
        async def route_handler(route, request):
            if _should_block_url(request.url):
                await route.abort()
            else:
                await route.continue_()

        page = await context.new_page()
        await page.route("**/*", route_handler)

        try:
            # Navigate with a generous timeout
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait a moment for any lazy-loaded content
            await asyncio.sleep(2)

            # Get page title
            title = await page.title()
            if not title or title.strip() in ("", "about:blank"):
                title = url

            # Get HTML content
            html_content = await page.content()

            # Generate PDF (high-fidelity for display)
            pdf_bytes = await page.pdf(
                format="A4",
                print_background=True,
                margin={"top": "2cm", "bottom": "2cm", "left": "2cm", "right": "2cm"}
            )

            return html_content, pdf_bytes, title

        except PlaywrightTimeout:
            logger.warning(f"Timeout loading {url}, returning partial content")
            html_content = await page.content()
            title = await page.title() or url
            pdf_bytes = await page.pdf(format="A4", print_background=True)
            return html_content, pdf_bytes, title

        finally:
            await context.close()
            await browser.close()


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
    Capture a webpage using Playwright and extract clean markdown.

    Pipeline:
    1. Playwright (Chromium) renders the page with JS, blocks trackers
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
        # Step 1: Render with Playwright
        html_content, pdf_bytes, title = await _render_page_with_playwright(url)

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
