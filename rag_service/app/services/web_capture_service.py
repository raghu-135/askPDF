import os
import hashlib
import logging
import json
from datetime import datetime
from typing import Optional

import trafilatura
import markdown
from weasyprint import HTML, CSS

logger = logging.getLogger(__name__)

# Base directories for shared web captures and PDFs
WEBPAGES_DIR = "/static/webpages"
STATIC_DIR = "/static"
os.makedirs(WEBPAGES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# CSS styling for clean PDF output
PDF_STYLES = """
@page {
    size: A4;
    margin: 2cm;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #333;
}

h1 {
    font-size: 18pt;
    font-weight: 600;
    color: #1a1a1a;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 0.3em;
    margin-top: 1.5em;
    margin-bottom: 0.8em;
}

h2 {
    font-size: 14pt;
    font-weight: 600;
    color: #2a2a2a;
    margin-top: 1.3em;
    margin-bottom: 0.6em;
}

h3 {
    font-size: 12pt;
    font-weight: 600;
    color: #3a3a3a;
    margin-top: 1.2em;
    margin-bottom: 0.5em;
}

p {
    margin-bottom: 0.8em;
    text-align: left;
}

a {
    color: #2563eb;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

ul, ol {
    margin-left: 1.5em;
    margin-bottom: 0.8em;
}

li {
    margin-bottom: 0.3em;
}

code {
    background-color: #f3f4f6;
    padding: 0.2em 0.4em;
    border-radius: 3px;
    font-family: "SF Mono", Monaco, Inconsolata, "Fira Code", Consolas, monospace;
    font-size: 0.9em;
}

pre {
    background-color: #f3f4f6;
    padding: 1em;
    border-radius: 6px;
    overflow-x: auto;
    margin-bottom: 1em;
}

pre code {
    background: none;
    padding: 0;
}

blockquote {
    border-left: 4px solid #e0e0e0;
    padding-left: 1em;
    margin-left: 0;
    color: #666;
    font-style: italic;
}

img {
    max-width: 100%;
    height: auto;
    margin: 1em 0;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 1em 0;
}

th, td {
    border: 1px solid #e0e0e0;
    padding: 0.5em;
    text-align: left;
}

th {
    background-color: #f9fafb;
    font-weight: 600;
}

hr {
    border: none;
    border-top: 1px solid #e0e0e0;
    margin: 2em 0;
}

/* Page header with source URL */
.page-header {
    font-size: 8pt;
    color: #666;
    border-bottom: 1px solid #e0e0e0;
    padding-bottom: 0.5em;
    margin-bottom: 1.5em;
}
"""


def _url_to_hash(url: str) -> str:
    """Generate a stable MD5 hash of a URL for file naming."""
    return hashlib.md5(url.encode()).hexdigest()


def _env_bool(name: str, default: bool) -> bool:
    """Parse boolean from environment variable."""
    value = os.getenv(name, str(default))
    return value.lower() == 'true'


def _markdown_to_pdf(markdown_content: str, url: str, title: str) -> bytes:
    """
    Convert markdown content to PDF bytes.

    Args:
        markdown_content: The markdown text to convert
        url: Source URL for the header
        title: Page title

    Returns:
        PDF as bytes
    """
    # Convert markdown to HTML
    html_content = markdown.markdown(
        markdown_content,
        extensions=['extra', 'codehilite', 'tables', 'fenced_code']
    )

    # Wrap with full HTML structure and header
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
</head>
<body>
    <div class="page-header">
        <strong>Source:</strong> {url}
    </div>
    {html_content}
</body>
</html>"""

    # Convert HTML to PDF using weasyprint
    pdf_bytes = HTML(string=full_html).write_pdf(
        stylesheets=[CSS(string=PDF_STYLES)]
    )

    return pdf_bytes


async def capture_webpage_as_pdf(url: str, force: bool = False) -> dict:
    """
    Capture a webpage and convert it to PDF for unified processing.

    Uses trafilatura to extract clean markdown from the webpage, then
    converts to PDF via weasyprint. The PDF is stored in the same location
    as uploaded PDFs (/static/{hash}.pdf) for display and reindexing purposes.
    A mapping file is also saved to track URL->PDF relationships for refresh operations.

    Args:
        url: The URL to capture
        force: If True, force recapture even if already cached

    Returns:
        dict with file_hash (PDF hash), url_hash, title, pdf_path, original_url, source_type
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

    # Fetch and extract markdown content using trafilatura
    downloaded = trafilatura.fetch_url(url)
    if downloaded is None:
        raise RuntimeError(f"Failed to fetch URL: {url}")

    # Extract clean markdown with configurable parameters from environment
    markdown_content = trafilatura.extract(
        downloaded,
        output_format='markdown',
        include_comments=_env_bool('TRAFILATURA_INCLUDE_COMMENTS', False),
        include_tables=_env_bool('TRAFILATURA_INCLUDE_TABLES', True),
        include_images=_env_bool('TRAFILATURA_INCLUDE_IMAGES', True),
        include_links=_env_bool('TRAFILATURA_INCLUDE_LINKS', True),
        include_formatting=_env_bool('TRAFILATURA_INCLUDE_FORMATTING', True),
        deduplicate=_env_bool('TRAFILATURA_DEDUPLICATE', True),
        favor_recall=_env_bool('TRAFILATURA_FAVOR_RECALL', True),
        fast=_env_bool('TRAFILATURA_FAST', False),
        url=url,
    )

    if not markdown_content:
        markdown_content = f"# {url}\n\nNo content could be extracted from this page."

    # Get title from the page metadata using bare_extraction
    metadata = trafilatura.bare_extraction(downloaded, only_with_metadata=True)
    if metadata and hasattr(metadata, 'title') and metadata.title:
        title = metadata.title
    else:
        title = url

    # Convert markdown to PDF
    pdf_bytes = _markdown_to_pdf(markdown_content, url, title)

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
