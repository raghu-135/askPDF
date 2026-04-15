import os
import hashlib
import logging
import json
import asyncio
import httpx
import time
from datetime import datetime
from typing import Optional
from pathlib import Path

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from markitdown import MarkItDown
from adblockparser import AdblockRules

logger = logging.getLogger(__name__)

# Base directories for shared web captures and PDFs
WEBPAGES_DIR = "/static/webpages"
STATIC_DIR = "/static"
os.makedirs(WEBPAGES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Common tracker/cookie banner domains to block

# Common cookie/consent banner button selectors to auto-accept/close
CONSENT_BUTTON_SELECTORS = [
    # High-priority specific selectors (fast path)
    "#onetrust-accept-btn-handler",  # OneTrust
    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",  # Cookiebot
    "#accept-recommended-btn-handler",  # OneTrust alternative
    ".fc-cta-consent",  # Funding Choices
    "button.fc-cta-consent",
    "[data-testid='accept-cookies']",
    "[data-testid='cookie-accept']",
    "[data-testid='consent-accept']",
    "[data-cookiebanner='accept']",
    "[data-action='accept']",
    "[data-action='accept-all']",
    # Modern privacy-focused sites (jina.ai, etc)
    "[class*='privacy'] button[class*='accept']",
    "[class*='privacy'] button[id*='accept']",
    "[class*='cookie'] button[class*='accept']",
    "[class*='cookie'] button[id*='accept']",
    "[class*='consent'] button[class*='accept']",
    "[class*='consent'] button[id*='accept']",
    "div[class*='banner'] button[class*='accept']",
    "div[class*='banner'] button[id*='accept']",
    "div[class*='popup'] button[class*='accept']",
    "div[class*='popup'] button[id*='accept']",
    "div[class*='modal'] button[class*='accept']",
    "div[class*='modal'] button[id*='accept']",
    "div[class*='overlay'] button[class*='accept']",
    "div[class*='overlay'] button[id*='accept']",
    # Shadow DOM piercing selectors (deeper penetration)
    "*:shadow(button[class*='accept'])",
    "*:shadow(button[id*='accept'])",
    "*:shadow([class*='accept-all'])",
    # Cookie consent buttons (general patterns)
    "button[id*='accept']",
    "button[class*='accept']",
    "button[aria-label*='accept' i]",
    "button[aria-label*='cookie' i]",
    "button[aria-label*='consent' i]",
    "button[data-testid*='accept']",
    "button[data-testid*='cookie']",
    "a[id*='accept']",
    "a[class*='accept']",
    # Generic cookie/consent class patterns
    ".cookie-banner .accept",
    ".cookie-banner .accept-all",
    ".cookie-consent .accept",
    ".cc-accept",  # Cookie Consent
    ".cc-allow",
    ".js-accept-cookies",
    ".accept-cookies",
    ".accept-all-cookies",
    "#accept-cookies",
    "#accept-all-cookies",
    # Generic text-based selectors (broader matching)
    "button:has-text('Accept')",
    "button:has-text('Accept all')",
    "button:has-text('Accept cookies')",
    "button:has-text('Allow')",
    "button:has-text('Allow all')",
    "button:has-text('Agree')",
    "button:has-text('I accept')",
    "button:has-text('Yes, accept')",
    "button:has-text('Yes, I accept')",
    "button:has-text('Got it')",
    "button:has-text('OK')",
    "button:has-text('Okay')",
    "button:has-text('Continue')",
    "button:has-text('I understand')",
    "button:has-text('Dismiss')",
    "button:has-text('Close')",
    "a:has-text('Accept')",
    "a:has-text('Accept all')",
    "a:has-text('Allow')",
    "a:has-text('Allow all')",
    "a:has-text('Agree')",
    "a:has-text('Continue')",
    "a:has-text('Dismiss')",
    "a:has-text('Close')",
    # Alternative paths: "Necessary only" buttons (often preferred)
    "button:has-text('Only necessary')",
    "button:has-text('Necessary only')",
    "button:has-text('Reject all')",
    "button:has-text('Decline')",
    "button:has-text('No, thanks')",
    "button:has-text('Dismiss')",
    "a:has-text('Only necessary')",
    "a:has-text('Reject all')",
    "[class*='reject']",
    "[class*='decline']",
    "[class*='necessary']",
    # Form/submit patterns
    "input[type='submit'][value*='accept' i]",
    "input[type='button'][value*='accept' i]",
    "input[type='submit'][value*='agree' i]",
    "input[type='button'][value*='agree' i]",
]

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

# Filter list configuration
FILTER_LIST_URL = "https://secure.fanboy.co.nz/fanboy-cookiemonster.txt"
FILTER_LIST_CACHE_PATH = "/tmp/fanboy-cookiemonster.txt"
FILTER_LIST_CACHE_DURATION = 86400  # 24 hours in seconds
_adblock_rules: Optional[AdblockRules] = None
_filter_list_last_update = 0


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


def _download_filter_list() -> bool:
    """Download filter list from remote URL and cache it locally."""
    global _filter_list_last_update
    try:
        logger.info(f"Downloading filter list from {FILTER_LIST_URL}")
        response = httpx.get(FILTER_LIST_URL, timeout=30)
        response.raise_for_status()

        # Write to cache
        Path(FILTER_LIST_CACHE_PATH).write_text(response.text, encoding='utf-8')
        _filter_list_last_update = time.time()
        logger.info(f"Filter list downloaded and cached to {FILTER_LIST_CACHE_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to download filter list: {e}")
        return False


def _load_filter_list() -> Optional[AdblockRules]:
    """Load filter list from cache or download if needed."""
    global _adblock_rules, _filter_list_last_update

    current_time = time.time()
    cache_exists = Path(FILTER_LIST_CACHE_PATH).exists()

    # Check if we need to download (cache expired or doesn't exist)
    if not cache_exists or (current_time - _filter_list_last_update > FILTER_LIST_CACHE_DURATION):
        if not _download_filter_list():
            logger.warning("Using existing cache or empty rules due to download failure")
            if not cache_exists:
                return None

    # Load from cache
    try:
        filter_text = Path(FILTER_LIST_CACHE_PATH).read_text(encoding='utf-8')
        raw_rules = []

        # Parse filter list (skip comments and empty lines)
        for line in filter_text.split('\n'):
            line = line.strip()
            # Skip comments, metadata, and empty lines
            if not line or line.startswith('!') or line.startswith('[Adblock'):
                continue
            # Skip exception rules (whitelisting)
            if line.startswith('@@'):
                continue
            raw_rules.append(line)

        # Create AdblockRules instance
        _adblock_rules = AdblockRules(raw_rules)
        logger.info(f"Loaded {len(raw_rules)} filter rules from {FILTER_LIST_CACHE_PATH}")
        return _adblock_rules
    except Exception as e:
        logger.error(f"Failed to load filter list: {e}")
        return None


def _should_block_with_filter_list(url: str) -> bool:
    """Check if URL should be blocked based on filter list rules."""
    global _adblock_rules

    # Load filter list if not already loaded
    if _adblock_rules is None:
        _adblock_rules = _load_filter_list()
        if _adblock_rules is None:
            return False

    try:
        # Check if URL should be blocked
        return _adblock_rules.should_block(url)
    except Exception as e:
        logger.warning(f"Error checking filter list for {url}: {e}")
        return False


async def _try_click_consent_button(page) -> bool:
    """
    Try to find and click a consent button on the page.

    Args:
        page: Playwright page object

    Returns:
        True if a consent button was clicked, False otherwise
    """
    for selector in CONSENT_BUTTON_SELECTORS:
        try:
            # Check if element exists and is visible
            element = page.locator(selector).first
            if element:
                # Check if element is visible (short timeout)
                try:
                    is_visible = await element.is_visible(timeout=50)
                    if not is_visible:
                        continue
                except Exception:
                    continue

                # Scroll element into view
                try:
                    await element.scroll_into_view_if_needed(timeout=500)
                except Exception:
                    pass  # Continue even if scroll fails

                # Click the accept button
                try:
                    await element.click(timeout=1000)
                    logger.info(f"Clicked consent button: {selector}")
                    # Wait briefly for banner to disappear/animate
                    await asyncio.sleep(0.3)
                    return True
                except Exception:
                    continue
        except Exception:
            # Element not found or not clickable, continue to next selector
            continue
    return False


async def _try_click_iframe_consent(page) -> bool:
    """
    Try to find and click consent buttons inside iframes (common for CMPs).

    Args:
        page: Playwright page object

    Returns:
        True if a consent button was clicked, False otherwise
    """
    try:
        # Get all iframes
        iframes = await page.locator("iframe").all()
        for iframe in iframes:
            try:
                # Check if iframe is visible
                if not await iframe.is_visible(timeout=50):
                    continue

                # Get frame content
                frame = await iframe.content_frame()
                if not frame:
                    continue

                # Try consent selectors in the iframe
                for selector in CONSENT_BUTTON_SELECTORS:
                    try:
                        element = frame.locator(selector).first
                        if not element:
                            continue

                        # Check visibility
                        try:
                            is_visible = await element.is_visible(timeout=50)
                            if not is_visible:
                                continue
                        except Exception:
                            continue

                        # Click the button
                        await element.click(timeout=1000)
                        logger.info(f"Clicked iframe consent button: {selector}")
                        await asyncio.sleep(0.3)
                        return True
                    except Exception:
                        continue
            except Exception:
                continue
    except Exception:
        pass
    return False


async def _accept_cookie_consent(page) -> bool:
    """
    Attempt to auto-accept cookie/consent banners on the page.
    Tries multiple times with increasing delays to catch late-loading banners.

    Args:
        page: Playwright page object

    Returns:
        True if a consent button was clicked, False otherwise
    """
    clicked = False

    # Try immediately first
    if await _try_click_consent_button(page):
        clicked = True

    # Try iframe consent if main page didn't work
    if not clicked:
        if await _try_click_iframe_consent(page):
            clicked = True

    # Multiple retry attempts with increasing delays
    # Banners often animate in or load after initial paint
    retry_delays = [0.5, 1.0, 1.5]

    for delay in retry_delays:
        if clicked:
            break

        await asyncio.sleep(delay)

        # Try main page again
        if await _try_click_consent_button(page):
            clicked = True
            break

        # Try iframes again
        if await _try_click_iframe_consent(page):
            clicked = True
            break

    if clicked:
        # Final wait for any animations/transitions
        await asyncio.sleep(0.5)

    return clicked


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
            # Check against blocked domains
            if _should_block_url(request.url):
                await route.abort()
                return

            # Check against filter list (cookie banners, ads, etc.)
            if _should_block_with_filter_list(request.url):
                logger.debug(f"Blocked by filter list: {request.url}")
                await route.abort()
                return

            await route.continue_()

        page = await context.new_page()
        await page.route("**/*", route_handler)

        try:
            # Navigate with a generous timeout
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Wait for lazy-loaded content (cookie banner handling has its own delays)
            await asyncio.sleep(1)

            # Attempt to auto-accept cookie/consent banners (with retry logic)
            await _accept_cookie_consent(page)

            # Get page title
            title = await page.title()
            if not title or title.strip() in ("", "about:blank"):
                title = url

            # Get HTML content
            html_content = await page.content()

            # Generate PDF (high-fidelity for display, no margins for continuous appearance)
            pdf_bytes = await page.pdf(
                format="A4",
                print_background=True,
                margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}
            )

            return html_content, pdf_bytes, title

        except PlaywrightTimeout:
            logger.warning(f"Timeout loading {url}, returning partial content")
            html_content = await page.content()
            title = await page.title() or url
            pdf_bytes = await page.pdf(format="A4", print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
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
