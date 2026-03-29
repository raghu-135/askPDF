import os
import re
import base64
import hashlib
import logging
from typing import Optional
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
import trafilatura

logger = logging.getLogger(__name__)

# Base directory for shared web captures
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


async def _fetch_bytes(client: httpx.AsyncClient, url: str) -> Optional[bytes]:
    try:
        r = await client.get(url, headers=_FETCH_HEADERS, follow_redirects=True, timeout=10.0)
        r.raise_for_status()
        if len(r.content) > _MAX_ASSET_BYTES:
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
    def replace(m: re.Match) -> str:
        inner = m.group(1).strip("'\"")
        if inner.startswith("data:") or inner.startswith("http"):
            return m.group(0)
        return f'url("{urljoin(stylesheet_url, inner)}")'
    return re.sub(r"url\(([^)]+)\)", replace, css_text)


def _url_to_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


async def capture_webpage(url: str, force: bool = False, webpages_dir: str = WEBPAGES_DIR) -> dict:
    """
    Fetch, inline assets, and save a self-contained HTML file.
    Also extracts clean text and metadata using Trafilatura for RAG.
    """
    file_hash = _url_to_hash(url)
    saved_path = os.path.join(webpages_dir, f"{file_hash}.html")

    # 1. Fetch the raw page content
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        resp = await client.get(url, headers=_FETCH_HEADERS)
        resp.raise_for_status()
        html_text = resp.text
        base_url = str(resp.url)

    # 2. Extract clean metadata and text using Trafilatura
    # This is we enhancement for better RAG and content hashing
    downloaded = trafilatura.fetch_url(url)
    trafilatura_result = trafilatura.extract(downloaded, output_format="json", include_comments=False, include_tables=True)
    
    import json
    metadata = {}
    clean_text = ""
    if trafilatura_result:
        try:
            res_json = json.loads(trafilatura_result)
            clean_text = res_json.get("text", "")
            metadata = {
                "title": res_json.get("title"),
                "author": res_json.get("author"),
                "date": res_json.get("date"),
                "description": res_json.get("description"),
                "sitename": res_json.get("sitename"),
            }
        except Exception as e:
            logger.warning(f"Trafilatura JSON parsing failed: {e}")

    # Fallback title if trafilatura fails
    soup = BeautifulSoup(html_text, "lxml")
    title = metadata.get("title") or (soup.title.get_text(strip=True) if soup.title else url)

    # 3. Content Hashing (using clean text for more stable hashing)
    if not clean_text:
        # Fallback to soup text if trafilatura missed everything
        body_text = soup.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in body_text.splitlines() if line.strip()]
        clean_text = "\n".join(lines)
        
    content_hash = hashlib.md5(clean_text.encode("utf-8", errors="replace")).hexdigest()

    # 4. If not forced and file exists, return early (avoid expensive inlining)
    if not force and os.path.exists(saved_path):
        return {
            "file_hash": file_hash,
            "title": title,
            "cached": True,
            "content_hash": content_hash,
            "metadata": metadata
        }

    # 5. Visual Inlining for UI (BeautifulSoup approach)
    # Inject <base>
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

    async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
        # Stylesheets
        for link in soup.find_all("link", rel=lambda r: r and "stylesheet" in r):
            href = link.get("href", "")
            if not href or href.startswith("data:"): continue
            abs_href = urljoin(base_url, href)
            css_bytes = await _fetch_bytes(client, abs_href)
            if css_bytes:
                css_text = css_bytes.decode("utf-8", errors="replace")
                css_text = _rewrite_css_urls(css_text, abs_href)
                style_tag = soup.new_tag("style")
                style_tag.string = css_text
                link.replace_with(style_tag)
            else:
                link["href"] = abs_href

        # Images
        for img in soup.find_all("img"):
            src = img.get("src", "")
            if not src or src.startswith("data:"): continue
            abs_src = urljoin(base_url, src)
            img_bytes = await _fetch_bytes(client, abs_src)
            if img_bytes:
                img["src"] = _data_uri(img_bytes, _image_mime(abs_src))
            else:
                img["src"] = abs_src

    # Remove external scripts
    for script in soup.find_all("script", src=True):
        script.decompose()

    # Save to shared volume
    with open(saved_path, "w", encoding="utf-8") as fh:
        fh.write(str(soup))

    return {
        "file_hash": file_hash,
        "title": title,
        "cached": False,
        "content_hash": content_hash,
        "metadata": metadata,
    }
