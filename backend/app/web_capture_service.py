"""
web_capture_service.py
------------------------
Fetches a webpage and produces a self-contained HTML file with:
  - External stylesheets inlined as <style> blocks
  - Images converted to base64 data URIs
  - A <base> tag so any remaining relative URLs resolve against the original site
  - External <script src> tags removed (they fail in sandboxed iframes anyway)
  - Inline <script> blocks kept intact

The resulting file is saved to /static/webpages/{file_hash}.html and served from
the backend, so the frontend can point an <iframe> at a same-origin URL with
no X-Frame-Options or CSP interference from the original site.

No Playwright / Chrome required — uses httpx + BeautifulSoup (already in stack).
"""

import os
import re
import base64
import hashlib
import logging
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

WEBPAGES_DIR = "/static/webpages"
os.makedirs(WEBPAGES_DIR, exist_ok=True)

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# Maximum bytes we'll try to inline per asset (2 MB)
_MAX_ASSET_BYTES = 2 * 1024 * 1024


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_bytes(client: httpx.AsyncClient, url: str) -> Optional[bytes]:
    """Fetch a URL returning raw bytes, or None on any error."""
    try:
        r = await client.get(url, headers=_FETCH_HEADERS, follow_redirects=True, timeout=10.0)
        r.raise_for_status()
        if len(r.content) > _MAX_ASSET_BYTES:
            logger.debug("Asset too large to inline, skipping: %s", url)
            return None
        return r.content
    except Exception as exc:
        logger.debug("Could not fetch asset %s: %s", url, exc)
        return None


def _data_uri(content: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(content).decode()}"


def _image_mime(url: str) -> str:
    ext = urlparse(url).path.lower().rsplit(".", 1)[-1]
    return {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "svg": "image/svg+xml",
        "ico": "image/x-icon",
        "avif": "image/avif",
        "bmp": "image/bmp",
    }.get(ext, "image/png")


def _rewrite_css_urls(css_text: str, stylesheet_url: str) -> str:
    """Make url() references inside a CSS string absolute."""
    def replace(m: re.Match) -> str:
        inner = m.group(1).strip("'\"")
        if inner.startswith("data:") or inner.startswith("http"):
            return m.group(0)
        return f'url("{urljoin(stylesheet_url, inner)}")'

    return re.sub(r"url\(([^)]+)\)", replace, css_text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def webpage_path(file_hash: str) -> str:
    return os.path.join(WEBPAGES_DIR, f"{file_hash}.html")


def webpage_exists(file_hash: str) -> bool:
    return os.path.exists(webpage_path(file_hash))


def url_to_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


async def capture_webpage(url: str, force: bool = False) -> dict:
    """
    Fetch *url*, inline assets, and save a self-contained HTML file.

    Args:
        url:   The page URL to capture.
        force: Re-fetch even if a cached copy already exists.

    Returns:
        {
          "file_hash": str,
          "title":     str,
          "saved_path": str,
          "cached":    bool,   # True if we returned an existing capture
        }
    """
    file_hash = url_to_hash(url)
    saved_path = webpage_path(file_hash)

    # Return cached copy if available and not forced
    if not force and os.path.exists(saved_path):
        with open(saved_path, "r", encoding="utf-8", errors="replace") as fh:
            cached_html = fh.read()
        soup_tmp = BeautifulSoup(cached_html, "lxml")
        title = soup_tmp.title.get_text(strip=True) if soup_tmp.title else url
        return {"file_hash": file_hash, "title": title, "saved_path": saved_path, "cached": True}

    # ---- Step 1: fetch the page ----------------------------------------
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        resp = await client.get(url, headers=_FETCH_HEADERS)
        resp.raise_for_status()
        html_text = resp.text
        base_url = str(resp.url)  # final URL after any redirects

    soup = BeautifulSoup(html_text, "lxml")
    title = soup.title.get_text(strip=True) if soup.title else url

    # ---- Step 2: inject <base> so un-inlined references still resolve ---
    for existing in soup.find_all("base"):
        existing.decompose()
    head = soup.find("head")
    if not head:
        head = soup.new_tag("head")
        if soup.html:
            soup.html.insert(0, head)
        else:
            soup.insert(0, head)
    base_tag = soup.new_tag("base", href=base_url)
    head.insert(0, base_tag)

    # ---- Step 3: inline assets -----------------------------------------
    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:

        # Stylesheets → <style> blocks
        for link in soup.find_all("link", rel=lambda r: r and "stylesheet" in r):
            href = link.get("href", "")
            if not href or href.startswith("data:"):
                continue
            abs_href = urljoin(base_url, href)
            css_bytes = await _fetch_bytes(client, abs_href)
            if css_bytes:
                css_text = css_bytes.decode("utf-8", errors="replace")
                css_text = _rewrite_css_urls(css_text, abs_href)
                style_tag = soup.new_tag("style")
                style_tag.string = css_text
                link.replace_with(style_tag)
            else:
                # Make the href absolute so the browser can try to load it
                link["href"] = abs_href

        # Images → base64 data URIs
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if not src or src.startswith("data:"):
                continue
            abs_src = urljoin(base_url, src)
            img_bytes = await _fetch_bytes(client, abs_src)
            if img_bytes:
                img["src"] = _data_uri(img_bytes, _image_mime(abs_src))
            else:
                img["src"] = abs_src  # absolute fallback

        # srcset attributes (picture / responsive images)
        for el in soup.find_all(srcset=True):
            parts = []
            for part in el["srcset"].split(","):
                part = part.strip()
                tokens = part.split()
                if tokens:
                    abs_src = urljoin(base_url, tokens[0])
                    img_bytes = await _fetch_bytes(client, abs_src)
                    if img_bytes:
                        tokens[0] = _data_uri(img_bytes, _image_mime(abs_src))
                    else:
                        tokens[0] = abs_src
                parts.append(" ".join(tokens))
            el["srcset"] = ", ".join(parts)

        # Favicon / touch icons
        for link in soup.find_all("link", rel=lambda r: r and any(
            v in r for v in ("icon", "apple-touch-icon", "shortcut")
        )):
            href = link.get("href", "")
            if href and not href.startswith("data:"):
                abs_href = urljoin(base_url, href)
                icon_bytes = await _fetch_bytes(client, abs_href)
                if icon_bytes:
                    link["href"] = _data_uri(icon_bytes, _image_mime(abs_href))
                else:
                    link["href"] = abs_href

    # ---- Step 4: remove external scripts (CORS fails in sandbox) --------
    for script in soup.find_all("script", src=True):
        script.decompose()

    # ---- Step 5: save ---------------------------------------------------
    final_html = str(soup)
    with open(saved_path, "w", encoding="utf-8") as fh:
        fh.write(final_html)

    logger.info("Captured webpage %s → %s", url, saved_path)
    return {"file_hash": file_hash, "title": title, "saved_path": saved_path, "cached": False}
