import os
import hashlib
import logging
import json
import asyncio
import httpx
import time
import subprocess
import tempfile
from datetime import datetime
from typing import Optional
from pathlib import Path

from markitdown import MarkItDown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Base directories for shared web captures and PDFs
WEBPAGES_DIR = "/static/webpages"
STATIC_DIR = "/static"
os.makedirs(WEBPAGES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Gotenberg configuration
GOTENBERG_SERVICE_URL = os.environ.get("GOTENBERG_SERVICE_URL", "http://gotenberg:3000")
COOKIES_DIR = os.environ.get("COOKIES_DIR", "/data/cookies")
os.makedirs(COOKIES_DIR, exist_ok=True)

# Default cookie file path (Netscape format)
DEFAULT_COOKIE_FILE = os.path.join(COOKIES_DIR, "default_cookies.txt")

# CSS selectors for cookie banner removal (applied to HTML post-capture)
COOKIE_BANNER_SELECTORS = [
    # OneTrust
    "#onetrust-consent-sdk",
    "#onetrust-banner-sdk",
    "#onetrust-pc-sdk",
    ".onetrust",
    # Cookiebot
    "#CybotCookiebotDialog",
    ".CybotCookiebotDialog",
    "#cookiebanner",
    ".cookiebanner",
    # Funding Choices
    ".fc-consent-root",
    ".fc-dialog",
    ".fc-cta-consent",
    # TrustArc
    ".trustarc-banner",
    "#truste-consent-track",
    ".truste_consent",
    # Quantcast
    ".qc-cmp2-container",
    "#qc-cmp2-container",
    # Generic patterns
    "[class*='cookie-banner']",
    "[class*='cookiebanner']",
    "[class*='cookie-consent']",
    "[class*='cookieconsent']",
    "[id*='cookie-banner']",
    "[id*='cookiebanner']",
    "[id*='cookie-consent']",
    "[id*='cookieconsent']",
    "[class*='consent-banner']",
    "[class*='consentbanner']",
    "[id*='consent-banner']",
    "[class*='gdpr-banner']",
    "[id*='gdpr-banner']",
    "[class*='privacy-banner']",
    "[id*='privacy-banner']",
    "[class*='cookie-popup']",
    "[id*='cookie-popup']",
    "[class*='cookie-modal']",
    "[id*='cookie-modal']",
    "[class*='cookie-overlay']",
    "[id*='cookie-overlay']",
    # Common CMP (Consent Management Platform) selectors
    "[data-testid*='cookie']",
    "[data-testid*='consent']",
    "[data-cookiebanner]",
    "[data-consent]",
    # Shadow host selectors (common for modern CMPs)
    "#cmp",
    ".cmp",
    "#cmp-banner",
    "[id*='cmp-']",
]


def _url_to_hash(url: str) -> str:
    """Generate a stable MD5 hash of a URL for file naming."""
    return hashlib.md5(url.encode()).hexdigest()


def _remove_cookie_banners_from_html(html_content: str) -> str:
    """
    Remove cookie banners and consent dialogs from HTML content using BeautifulSoup.
    
    Args:
        html_content: Raw HTML string
        
    Returns:
        Cleaned HTML string with cookie banners removed
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        removed_count = 0
        for selector in COOKIE_BANNER_SELECTORS:
            try:
                # Try CSS selector
                elements = soup.select(selector)
                for element in elements:
                    element.decompose()
                    removed_count += 1
            except Exception:
                # Invalid selector, skip
                continue
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} cookie banner elements from HTML")
        
        return str(soup)
    except Exception as e:
        logger.warning(f"Failed to remove cookie banners: {e}")
        return html_content


async def _capture_with_singlefile(
    url: str, 
    cookie_file: Optional[str] = None,
    timeout: int = 60
) -> tuple[str, str]:
    """
    Capture a webpage using SingleFile CLI via subprocess.
    Returns full-fidelity HTML with all resources embedded.
    
    Pipeline:
    1. Call SingleFile CLI to capture URL to HTML file
    2. Read the HTML file
    3. Apply post-processing (cookie banner removal)
    4. Return cleaned HTML and extracted title
    
    Args:
        url: The URL to capture
        cookie_file: Optional path to Netscape format cookies.txt file
        timeout: Maximum time to wait for capture (seconds)
        
    Returns:
        tuple of (html_content: str, title: str)
    """
    # Get temporary file path (don't create the file yet - SingleFile will create it)
    output_path = os.path.join(tempfile.gettempdir(), f"singlefile_{hashlib.md5(url.encode()).hexdigest()[:8]}.html")
    
    # Build SingleFile CLI command as shell string
    # single-file uses Chromium via Chrome DevTools Protocol
    url_escaped = url.replace('"', '\\"')
    output_escaped = output_path.replace('"', '\\"')
    
    cmd_str = f'npx single-file "{url_escaped}" "{output_escaped}" --browser-executable-path=/usr/bin/chromium --browser-headless=true --dump-content=false --remove-hidden-elements=true --remove-unused-styles=true --remove-unused-fonts=true --compress-html=true'
    
    # Add cookie file if provided
    if cookie_file and os.path.exists(cookie_file):
        cookie_escaped = cookie_file.replace('"', '\\"')
        cmd_str += f' --browser-cookies-file="{cookie_escaped}"'
        logger.info(f"Using cookie file: {cookie_file}")
    
    try:
        logger.info(f"Capturing webpage with SingleFile CLI: {url}")
        logger.info(f"Command: {cmd_str}")
        
        # Run SingleFile CLI as subprocess with shell=True
        process = await asyncio.create_subprocess_shell(
            cmd_str,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024*1024  # 1MB buffer
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.error(f"SingleFile timeout capturing {url}")
            raise RuntimeError(f"Timeout capturing webpage (>{timeout}s)")
        
        # Log process results for debugging
        stdout_text = stdout.decode('utf-8', errors='ignore')[:500] if stdout else ""
        stderr_text = stderr.decode('utf-8', errors='ignore')[:500] if stderr else ""
        logger.info(f"SingleFile exit code: {process.returncode}")
        if stdout_text:
            logger.info(f"SingleFile stdout: {stdout_text}")
        if stderr_text:
            logger.info(f"SingleFile stderr: {stderr_text}")
        
        if process.returncode != 0:
            logger.error(f"SingleFile failed with exit code {process.returncode}")
            logger.error(f"stderr: {stderr_text}")
            raise RuntimeError(f"SingleFile capture failed: {stderr_text}")
        
        # Read the captured HTML file
        if not os.path.exists(output_path):
            logger.error(f"SingleFile did not create output file at {output_path}")
            raise RuntimeError("SingleFile did not create output file")
        
        file_size = os.path.getsize(output_path)
        logger.debug(f"SingleFile output file size: {file_size} bytes")
        
        with open(output_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        content_len = len(html_content) if html_content else 0
        logger.debug(f"SingleFile HTML content length: {content_len} chars")
        
        if not html_content or len(html_content) < 100:
            raise RuntimeError(f"SingleFile returned empty or invalid HTML content (length: {content_len})")
        
        # Extract title from HTML
        title = url
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                title = title_tag.string.strip()
        except Exception:
            pass
        
        logger.info(f"SingleFile captured {url} successfully (title: {title[:50]}...)")
        
        # Post-process: remove cookie banners
        html_content = _remove_cookie_banners_from_html(html_content)
        
        return html_content, title
        
    except Exception as e:
        logger.error(f"SingleFile capture failed for {url}: {e}")
        raise RuntimeError(f"Failed to capture webpage: {str(e)}")
    finally:
        # Clean up temp file
        try:
            if os.path.exists(output_path):
                os.unlink(output_path)
        except Exception:
            pass


async def _convert_html_to_pdf_with_gotenberg(
    html_content: str,
    timeout: int = 60
) -> bytes:
    """
    Convert HTML content to PDF using Gotenberg service.
    
    Args:
        html_content: HTML string to convert
        timeout: Maximum time to wait for conversion (seconds)
        
    Returns:
        PDF content as bytes
    """
    # Create temporary file for HTML content
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
        tmp.write(html_content)
        tmp_path = tmp.name
    
    try:
        async with httpx.AsyncClient() as client:
            logger.info("Converting HTML to PDF with Gotenberg")
            
            # Gotenberg expects files in multipart/form-data
            with open(tmp_path, 'rb') as f:
                files = {'files': ('index.html', f, 'text/html')}
                response = await client.post(
                    f"{GOTENBERG_SERVICE_URL}/forms/chromium/convert/html",
                    files=files,
                    timeout=timeout
                )
            
            response.raise_for_status()
            pdf_bytes = response.content
            
            logger.info(f"Gotenberg generated PDF ({len(pdf_bytes)} bytes)")
            return pdf_bytes
            
    except httpx.TimeoutException:
        logger.error("Gotenberg timeout converting HTML to PDF")
        raise RuntimeError(f"Timeout converting to PDF (>{timeout}s)")
    except Exception as e:
        logger.error(f"Gotenberg PDF conversion failed: {e}")
        raise RuntimeError(f"Failed to convert to PDF: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
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


async def capture_webpage(
    url: str, 
    force: bool = False,
    cookie_file: Optional[str] = None
) -> dict:
    """
    Capture a webpage using SingleFile + Gotenberg pipeline and extract clean markdown.

    Pipeline:
    1. SingleFile captures full-fidelity HTML with embedded resources
    2. Post-process HTML to remove cookie banners
    3. Gotenberg converts HTML to PDF
    4. MarkItDown extracts clean Markdown from HTML
    5. Save PDF to /static/{hash}.pdf
    6. Save mapping file for URL→PDF lookup

    Args:
        url: The URL to capture
        force: If True, force recapture even if already cached
        cookie_file: Optional path to Netscape format cookies.txt file for authenticated pages

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
        # Step 1: Capture with SingleFile (full-fidelity HTML)
        html_content, title = await _capture_with_singlefile(url, cookie_file=cookie_file)

        # Step 2: Convert HTML to PDF with Gotenberg
        pdf_bytes = await _convert_html_to_pdf_with_gotenberg(html_content)

        # Step 3: Extract clean markdown
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
