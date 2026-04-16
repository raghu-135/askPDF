"""
Web Capture Service - Gotenberg-based implementation

Uses Gotenberg's Chromium for PDF generation, with simple HTTP fetching for HTML content.
"""

import os
import hashlib
import logging
import json
import re
import base64
from datetime import datetime
from io import BytesIO
from typing import Optional, List

import httpx
from markitdown import MarkItDown

from app.services.cookie_presets import get_consent_cookies_for_url, cookies_to_json, Cookie

logger = logging.getLogger(__name__)

# Base directories for shared web captures and PDFs
WEBPAGES_DIR = "/static/webpages"
STATIC_DIR = "/static"
os.makedirs(WEBPAGES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Gotenberg configuration
GOTENBERG_URL = os.getenv("GOTENBERG_URL", "http://gotenberg:3000")


def _url_to_hash(url: str) -> str:
    """Generate a stable MD5 hash of a URL for file naming."""
    return hashlib.md5(url.encode()).hexdigest()


def _cookies_to_header_string(cookies: List[Cookie]) -> str:
    """Convert cookies to a Cookie header string format."""
    return "; ".join([f"{c.name}={c.value}" for c in cookies])


# Banner suppression CSS to hide common cookie banners via emulateScreenMediaType
deviceType = "screen"


async def _convert_url_to_pdf_with_gotenberg(
    url: str,
    custom_cookies: Optional[List[Cookie]] = None,
    wait_delay: str = "4s",
) -> bytes:
    """
    Convert URL to PDF using Gotenberg's /forms/chromium/convert/url endpoint.
    
    Uses a 4-layer approach to suppress cookie banners:
    1. Proper domain-matched cookies via 'cookies' parameter
    2. Cookie header via 'extraHttpHeaders' as backup
    3. JavaScript suppression via 'waitForExpression' (sets localStorage, hides banners)
    4. Extended wait delay for banner animations
    
    Args:
        url: The URL to convert
        custom_cookies: Optional list of custom Cookie objects to use instead of presets
        wait_delay: Duration to wait before capturing (default: "4s")
        
    Returns:
        PDF bytes
        
    Raises:
        RuntimeError: If PDF generation fails
    """
    try:
        async with httpx.AsyncClient() as client:
            # Gotenberg requires multipart/form-data encoding
            # We use files= with BytesIO to force multipart encoding for form fields
            
            # Get consent cookies - use custom if provided, otherwise use presets
            if custom_cookies is not None:
                cookies = custom_cookies
                logger.info(f"Using {len(cookies)} custom cookies for {url}")
            else:
                cookies = get_consent_cookies_for_url(url)
                logger.info(f"Using {len(cookies)} preset consent cookies for {url}")
            
            # Convert cookies to Gotenberg's JSON format
            cookies_json = cookies_to_json(cookies)
            
            # Layer 2: Also send as Cookie header via extraHttpHeaders
            cookie_header = _cookies_to_header_string(cookies)
            extra_headers = {"Cookie": cookie_header}
            
            # Layer 3: Inject JavaScript to hide banners and set localStorage
            # Uses extraScriptTags to inject early in page load
            banner_suppression_script = """document.addEventListener('DOMContentLoaded',function(){try{localStorage.setItem('uc_settings','{"isValidConsent":true,"isFirstVisit":false}'),localStorage.setItem('uc_user_interaction','true'),localStorage.setItem('OptanonConsent','true'),localStorage.setItem('OptanonAlertBoxClosed','true'),localStorage.setItem('CookieConsent','{"necessary":true,"marketing":true}'),localStorage.setItem('cookieyes-consent','yes'),localStorage.setItem('cookie_consent','accepted');var e=document.createElement('style');e.textContent='#usercentrics-root,#onetrust-consent-sdk,#CybotCookiebotDialog,.cookie-banner,.cookie-consent,#gdpr-banner,#osano-cm-manage-dialog,.cc-window,.cc-banner,.didomi-popup-container,#sp-cc,.borlabs-cookie-preference,.iubenda-cs-container,.moove_gdpr_cookie_info_bar{display:none!important;visibility:hidden!important;opacity:0!important}',document.head.appendChild(e)}catch(e){}});"""
            
            script_b64 = base64.b64encode(banner_suppression_script.encode()).decode()
            extra_scripts = [{"src": f"data:text/javascript;base64,{script_b64}"}]
            
            form_fields = {
                "url": (None, url),
                "paperWidth": (None, "8.27"),
                "paperHeight": (None, "11.69"),
                "marginTop": (None, "0"),
                "marginBottom": (None, "0"),
                "marginLeft": (None, "0"),
                "marginRight": (None, "0"),
                "printBackground": (None, "true"),
                "preferCssPageSize": (None, "false"),
                "cookies": (None, cookies_json),
                "extraHttpHeaders": (None, json.dumps(extra_headers)),
                "waitDelay": (None, wait_delay),
                "emulatedMediaType": (None, "screen"),
                "extraScriptTags": (None, json.dumps(extra_scripts)),
            }
            
            logger.info(f"PDF conversion request: url={url}, cookies={len(cookies)}, waitDelay={wait_delay}")
            
            response = await client.post(
                f"{GOTENBERG_URL}/forms/chromium/convert/url",
                files=form_fields,
                timeout=60.0
            )
            response.raise_for_status()
            logger.info(f"Successfully generated PDF for {url} ({len(response.content)} bytes)")
            return response.content
    except httpx.HTTPError as e:
        logger.error(f"Gotenberg HTTP error for {url}: {e}")
        raise RuntimeError(f"PDF generation failed: HTTP error - {e}")
    except Exception as e:
        logger.error(f"Failed to generate PDF for {url}: {e}", exc_info=True)
        raise RuntimeError(f"PDF generation failed: {e}")


async def _fetch_html_and_title(url: str) -> tuple[str, str]:
    """
    Fetch HTML content and extract title from URL.
    
    Args:
        url: The URL to fetch
        
    Returns:
        tuple of (html_content: str, title: str)
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            html_content = response.text
            
            # Extract title from HTML
            title = url  # fallback
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
            if title_match:
                title = title_match.group(1).strip()
                # Remove extra whitespace
                title = re.sub(r'\s+', ' ', title)
            
            return html_content, title
    except Exception as e:
        logger.warning(f"Failed to fetch HTML for {url}: {e}")
        # Return minimal HTML and URL as title
        return f"<html><body><p>Failed to retrieve content from {url}</p></body></html>", url


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


async def capture_webpage(
    url: str,
    force: bool = False,
    custom_cookies: Optional[List[Cookie]] = None,
    wait_delay: str = "4s",
) -> dict:
    """
    Capture a webpage using Gotenberg for PDF and simple HTTP fetch for HTML.

    Pipeline:
    1. Gotenberg converts URL to PDF (uses its built-in Chromium)
    2. Fetch HTML separately for markdown extraction
    3. HTML → MarkItDown → clean Markdown (for vector DB)
    4. Save PDF to /static/{hash}.pdf
    5. Save mapping file for URL→PDF lookup

    Uses preset consent cookies to suppress cookie banners by default.

    Args:
        url: The URL to capture
        force: If True, force recapture even if already cached
        custom_cookies: Optional list of custom Cookie objects to override presets
        wait_delay: Duration to wait for cookie banners to settle (default: "4s")

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
    logger.info(f"Capturing webpage: {url} (wait_delay={wait_delay})")

    try:
        # Step 1: Get PDF from Gotenberg (runs Chromium internally)
        pdf_bytes = await _convert_url_to_pdf_with_gotenberg(
            url,
            custom_cookies=custom_cookies,
            wait_delay=wait_delay,
        )
        
        # Step 2: Fetch HTML separately for markdown extraction
        html_content, title = await _fetch_html_and_title(url)

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
async def capture_webpage_as_pdf(
    url: str,
    force: bool = False,
    custom_cookies: Optional[List[Cookie]] = None,
    wait_delay: str = "4s",
) -> dict:
    """Backward-compatible alias for capture_webpage."""
    return await capture_webpage(url, force, custom_cookies, wait_delay)


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
