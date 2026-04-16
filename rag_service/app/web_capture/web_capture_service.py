"""
Web Capture Service - Gotenberg-based implementation

Uses Gotenberg's Chromium for PDF generation, with simple HTTP fetching for HTML content.
"""

import os
import hashlib
import logging
import json
import re
from datetime import datetime
from typing import Optional, List

import httpx
from markitdown import MarkItDown

from app.web_capture.cookie_presets import Cookie
from app.web_capture.cmp_registry import detect_cmp, CMPType
from app.web_capture.capture_strategy import (
    CaptureOptions,
    ConsentStrategy,
    RenderStrategy,
    SuppressionValidator,
)

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


async def _convert_url_to_pdf_with_gotenberg(
    url: str,
    custom_cookies: Optional[List[Cookie]] = None,
    wait_delay: str = "2s",
    capture_options: Optional[CaptureOptions] = None,
    cmp_type: CMPType = CMPType.NONE,
    fallback: bool = False,
) -> bytes:
    """
    Convert URL to PDF using Gotenberg's /forms/chromium/convert/url endpoint.
    
    Uses a modular strategy-based approach to suppress cookie banners:
    1. Provider-specific cookies via 'cookies' parameter (proper domain/path semantics)
    2. Provider-specific localStorage via immediate-execution JavaScript
    3. Provider-specific CSS suppression via immediate-execution JavaScript
    4. Adaptive wait (expression-based for CMPs, delay-based otherwise)
    
    Args:
        url: The URL to convert
        custom_cookies: Optional list of custom Cookie objects to use instead of presets
        wait_delay: Duration to wait before capturing (default: "2s")
        capture_options: Optional CaptureOptions for advanced configuration
        cmp_type: The detected CMP type for provider-specific handling
        fallback: If True, use broader fallback strategy for retry attempts
        
    Returns:
        PDF bytes
        
    Raises:
        RuntimeError: If PDF generation fails
    """
    try:
        async with httpx.AsyncClient() as client:
            # Initialize or use provided capture options
            if capture_options is None:
                capture_options = CaptureOptions(wait_delay=wait_delay)
            
            # Get consent cookies using strategy
            if custom_cookies is not None:
                cookies = custom_cookies
                logger.info(f"Using {len(cookies)} custom cookies for {url}")
            else:
                cookies = ConsentStrategy.get_cookies(url, cmp_type, fallback=fallback)
                logger.info(f"Using {len(cookies)} strategy cookies for {cmp_type.value or 'unknown'} CMP")
            
            # Generate injection script for localStorage and CSS suppression
            injection_script = ConsentStrategy.get_injection_script(cmp_type, fallback=fallback)
            
            # Build Gotenberg options using render strategy
            form_fields = RenderStrategy.build_gotenberg_options(
                capture_options=capture_options,
                cmp_type=cmp_type,
                cookies=cookies,
                injection_script=injection_script,
                fallback=fallback,
            )
            
            # Add URL field
            form_fields["url"] = (None, url)
            
            logger.info(
                f"PDF conversion request: url={url}, cmp={cmp_type.value or 'none'}, "
                f"cookies={len(cookies)}, fallback={fallback}"
            )
            
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
    capture_options: Optional[CaptureOptions] = None,
) -> dict:
    """
    Capture a webpage using Gotenberg for PDF and simple HTTP fetch for HTML.

    Pipeline:
    1. Detect CMP type from URL/HTML
    2. Attempt 1: minimal intervention (provider-specific cookies, CSS, localStorage)
    3. Validate banner suppression in returned HTML
    4. If banner detected: Attempt 2 with broader fallback strategy
    5. Fetch HTML separately for markdown extraction
    6. HTML → MarkItDown → clean Markdown (for vector DB)
    7. Save PDF to /static/{hash}.pdf
    8. Save mapping file for URL→PDF lookup

    Uses adaptive consent handling with provider-specific strategies.

    Args:
        url: The URL to capture
        force: If True, force recapture even if already cached
        custom_cookies: Optional list of custom Cookie objects to override presets
        wait_delay: Duration to wait for cookie banners to settle (default: "4s")
        capture_options: Optional CaptureOptions for advanced configuration

    Returns:
        dict with file_hash (PDF hash), url_hash, title, pdf_path, original_url,
        source_type, markdown_content, cached flag, and retry_attempts
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
                    "retry_attempts": 0,
                }
        except Exception as e:
            logger.warning(f"Failed to read cached mapping for {url}: {e}")

    # Initialize capture options with backward-compatible defaults
    if capture_options is None:
        capture_options = CaptureOptions(wait_delay=wait_delay)
    
    # Determine max retries from options
    max_retries = capture_options.max_retries

    # Capture webpage
    logger.info(f"Capturing webpage: {url} (wait_delay={capture_options.wait_delay}, max_retries={max_retries})")

    try:
        # Step 1: Detect CMP type from URL first (fast path)
        cmp_type = detect_cmp(url, html_content=None)
        logger.info(f"Initial CMP detection from URL: {cmp_type.value or 'none'}")
        
        # Step 2: Attempt 1 - minimal intervention with detected CMP
        pdf_bytes = None
        html_content = None
        title = url
        retry_attempts = 0
        
        for attempt in range(1, max_retries + 2):  # +2 because range is exclusive and we start at 1
            is_fallback = attempt > 1
            
            # Re-detect CMP from HTML on retry if we have it
            if is_fallback and html_content:
                cmp_type = detect_cmp(url, html_content)
                logger.info(f"Retry {attempt - 1}: CMP re-detected from HTML: {cmp_type.value or 'none'}")
            
            logger.info(f"Capture attempt {attempt}: fallback={is_fallback}, cmp={cmp_type.value or 'none'}")
            
            # Generate PDF
            pdf_bytes = await _convert_url_to_pdf_with_gotenberg(
                url,
                custom_cookies=custom_cookies,
                wait_delay=capture_options.wait_delay,
                capture_options=capture_options,
                cmp_type=cmp_type,
                fallback=is_fallback,
            )
            
            # Step 3: Fetch HTML for validation and markdown extraction
            html_content, title = await _fetch_html_and_title(url)
            
            # Validate banner suppression (if not custom cookies)
            if custom_cookies is None:
                is_banner_present, confidence = SuppressionValidator.validate_banner_suppression(html_content)
                logger.info(f"Banner validation: present={is_banner_present}, confidence={confidence:.2f}")
                
                # Check if we should retry
                if SuppressionValidator.should_retry(confidence, attempt, max_retries):
                    retry_attempts = attempt
                    logger.warning(f"Banner detected (confidence={confidence:.2f}), will retry with broader strategy")
                    continue
                else:
                    # No retry needed or max retries reached
                    break
            else:
                # Custom cookies provided, skip validation
                break
        
        # Step 4: Extract clean markdown
        markdown_content = _extract_clean_markdown(html_content, url)

        # Step 5: Calculate PDF hash and save
        pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
        pdf_path = os.path.join(STATIC_DIR, f"{pdf_hash}.pdf")

        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        # Step 6: Save mapping file
        mapping = {
            "url": url,
            "pdf_hash": pdf_hash,
            "pdf_path": pdf_path,
            "title": title,
            "markdown_content": markdown_content,
            "captured_at": datetime.utcnow().isoformat(),
            "cmp_detected": cmp_type.value,
            "retry_attempts": retry_attempts,
        }
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2)

        logger.info(f"Captured {url} to {pdf_path} (hash: {pdf_hash}, retries: {retry_attempts})")

        return {
            "file_hash": pdf_hash,
            "url_hash": url_hash,
            "title": title,
            "pdf_path": pdf_path,
            "original_url": url,
            "source_type": "web",
            "markdown_content": markdown_content,
            "cached": False,
            "retry_attempts": retry_attempts,
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
    capture_options: Optional[CaptureOptions] = None,
) -> dict:
    """Backward-compatible alias for capture_webpage."""
    return await capture_webpage(url, force, custom_cookies, wait_delay, capture_options)


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
