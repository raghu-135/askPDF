import os
import hashlib
import logging
import json
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Base directories for shared web captures and PDFs
WEBPAGES_DIR = "/static/webpages"
STATIC_DIR = "/static"
os.makedirs(WEBPAGES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


def _url_to_hash(url: str) -> str:
    """Generate a stable MD5 hash of a URL for file naming."""
    return hashlib.md5(url.encode()).hexdigest()


async def capture_webpage_as_pdf(url: str, force: bool = False) -> dict:
    """
    Capture a webpage and convert it to PDF for unified processing.

    Uses Playwright to render the page and convert to PDF. The PDF is stored
    in the same location as uploaded PDFs (/static/{hash}.pdf) for display
    and reindexing purposes. A mapping file is also saved to track URL->PDF
    relationships for refresh operations.

    Args:
        url: The URL to capture
        force: If True, force recapture even if already cached

    Returns:
        dict with file_hash (PDF hash), url_hash, title, pdf_path, original_url, source_type
    """
    from playwright.async_api import async_playwright

    url_hash = _url_to_hash(url)
    mapping_path = os.path.join(WEBPAGES_DIR, f"{url_hash}.mapping.json")

    # Check if we have a cached mapping and PDF
    if not force and os.path.exists(mapping_path):
        try:
            with open(mapping_path, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            pdf_path = mapping.get("pdf_path")
            if pdf_path and os.path.exists(pdf_path):
                logger.info(f"Using cached PDF for {url}")
                return {
                    "file_hash": mapping["pdf_hash"],
                    "url_hash": url_hash,
                    "title": mapping["title"],
                    "pdf_path": pdf_path,
                    "original_url": url,
                    "source_type": "pdf",
                    "cached": True,
                }
        except Exception as e:
            logger.warning(f"Failed to read cached mapping for {url}: {e}")

    # Capture and convert to PDF
    logger.info(f"Capturing webpage to PDF: {url}")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        try:
            # Navigate and wait for network idle
            await page.goto(url, wait_until="networkidle", timeout=30000)

            # Extract title
            title = await page.title()
            if not title or title.strip() == "":
                title = url

            # Get HTML content for metadata extraction
            html_content = await page.content()

            # Generate PDF
            pdf_bytes = await page.pdf(
                format='A4',
                print_background=True,
                margin={'top': '20px', 'right': '20px', 'bottom': '20px', 'left': '20px'}
            )

        finally:
            await browser.close()

    # Calculate PDF hash and save
    pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
    pdf_path = os.path.join(STATIC_DIR, f"{pdf_hash}.pdf")

    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # Save mapping file for URL->PDF lookup (for refresh operations)
    mapping = {
        "url": url,
        "pdf_hash": pdf_hash,
        "pdf_path": pdf_path,
        "title": title,
        "captured_at": datetime.utcnow().isoformat(),
    }
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)

    logger.info(f"Captured {url} to PDF: {pdf_path} (hash: {pdf_hash})")

    return {
        "file_hash": pdf_hash,
        "url_hash": url_hash,
        "title": title,
        "pdf_path": pdf_path,
        "original_url": url,
        "source_type": "pdf",
        "cached": False,
    }


async def get_webpage_pdf_by_url_hash(url_hash: str) -> Optional[dict]:
    """
    Look up a webpage's PDF by its URL hash.
    Used for refresh operations to find the current PDF before recapturing.

    Args:
        url_hash: The MD5 hash of the URL

    Returns:
        dict with pdf_hash, pdf_path, url, title or None if not found
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
        }
    except Exception as e:
        logger.warning(f"Failed to read mapping for {url_hash}: {e}")
        return None
