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
from io import BytesIO
from typing import Optional

import httpx
from markitdown import MarkItDown

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


async def _convert_url_to_pdf_with_gotenberg(url: str) -> bytes:
    """
    Convert URL to PDF using Gotenberg's /forms/chromium/convert/url endpoint.
    
    Args:
        url: The URL to convert
        
    Returns:
        PDF bytes
        
    Raises:
        RuntimeError: If PDF generation fails
    """
    try:
        async with httpx.AsyncClient() as client:
            # Gotenberg requires multipart/form-data encoding
            # We use files= with BytesIO to force multipart encoding for form fields
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
            }
            response = await client.post(
                f"{GOTENBERG_URL}/forms/chromium/convert/url",
                files=form_fields,
                timeout=60.0
            )
            response.raise_for_status()
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


async def capture_webpage(url: str, force: bool = False) -> dict:
    """
    Capture a webpage using Gotenberg for PDF and simple HTTP fetch for HTML.

    Pipeline:
    1. Gotenberg converts URL to PDF (uses its built-in Chromium)
    2. Fetch HTML separately for markdown extraction
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
        # Step 1: Get PDF from Gotenberg (runs Chromium internally)
        pdf_bytes = await _convert_url_to_pdf_with_gotenberg(url)
        
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
