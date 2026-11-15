"""
Browser Capture API - Direct CDP Implementation
Simple FastAPI service that captures the current browser page as PDF using direct Chrome DevTools Protocol.
"""

import os
import hashlib
import base64
import json
import logging
import re
import asyncio
from contextlib import asynccontextmanager
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from url_policy import BrowserUrlPolicyError, validate_navigation_url
from cdp_desktop import cdp_debug_reachable
from pelorus_client import (
    CDP_FALLBACK_NOTE,
    DESKTOP_CONTROL_UNAVAILABLE_MESSAGE,
    desktop_screenshot,
    pelorus_get,
    pelorus_post,
    pelorus_ready,
)
import requests
import websockets
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, NameObject
from weasyprint import HTML, CSS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def _capture_service_lifespan(_app: FastAPI):
    for delay in (0, 2, 5):
        if delay:
            await asyncio.sleep(delay)
        try:
            await navigate_active_tab("https://example.com")
            logger.info("Browser warmup navigation complete")
            break
        except Exception as exc:
            logger.warning("Browser warmup navigate failed (delay=%ss): %s", delay, exc)
    yield


app = FastAPI(title="Browser Capture Service - Direct CDP", lifespan=_capture_service_lifespan)

# Thread pool for CPU-bound operations (WeasyPrint)
_thread_pool = ThreadPoolExecutor(max_workers=4)

CAPTURES_DIR = Path(os.environ.get("CAPTURES_DIR", "/captures"))
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

BROWSER_DEBUG_URL = os.environ.get("BROWSER_DEBUG_URL", "localhost:9222")
FILE_HASH_PATTERN = re.compile(r"^[a-f0-9]{32}$")
CAPTURE_READY_TIMEOUT_MS = int(os.environ.get("CAPTURE_READY_TIMEOUT_MS", "8000"))

# HTTP session with connection pooling for CDP requests
_cdp_session = requests.Session()
_cdp_session.headers.update({
    "Connection": "keep-alive",
    "Accept": "application/json"
})

_AGENT_TAB_TARGET_ID: str | None = None
_SYSTEM_URL_PREFIXES = ("chrome://", "chrome-extension://", "brave://", "about:")


def _fetch_debug_tabs() -> list[dict[str, Any]]:
    response = _cdp_session.get(f"http://{BROWSER_DEBUG_URL}/json", timeout=5)
    if response.status_code != 200:
        raise Exception("Failed to get tabs from debug interface")
    tabs = response.json()
    return tabs if isinstance(tabs, list) else []


def _is_http_content_tab(tab: Mapping[str, Any]) -> bool:
    url = str(tab.get("url") or "")
    return url.startswith(("http://", "https://"))


def _close_debug_tab(target_id: str) -> None:
    try:
        _cdp_session.get(f"http://{BROWSER_DEBUG_URL}/json/close/{target_id}", timeout=5)
    except Exception as exc:
        logger.warning("Failed to close browser tab %s: %s", target_id, exc)


async def ensure_agent_tab() -> dict[str, Any]:
    """Keep one agent-controlled content tab per capture service process."""
    global _AGENT_TAB_TARGET_ID
    tabs = _fetch_debug_tabs()
    if not tabs:
        raise Exception("No tabs available")
    content_tabs = [tab for tab in tabs if _is_http_content_tab(tab)]
    if _AGENT_TAB_TARGET_ID:
        if not any(str(tab.get("id")) == _AGENT_TAB_TARGET_ID for tab in tabs):
            _AGENT_TAB_TARGET_ID = None
    if _AGENT_TAB_TARGET_ID is None:
        fallback = next(
            (tab for tab in tabs if not str(tab.get("url") or "").startswith(_SYSTEM_URL_PREFIXES)),
            tabs[0],
        )
        chosen = content_tabs[0] if content_tabs else fallback
        _AGENT_TAB_TARGET_ID = str(chosen.get("id"))
    for tab in content_tabs:
        tab_id = str(tab.get("id") or "")
        if tab_id and tab_id != _AGENT_TAB_TARGET_ID:
            _close_debug_tab(tab_id)
    tabs = _fetch_debug_tabs()
    agent = next((tab for tab in tabs if str(tab.get("id")) == _AGENT_TAB_TARGET_ID), None)
    if agent is None:
        _AGENT_TAB_TARGET_ID = None
        return await ensure_agent_tab()
    logger.info("Agent tab: %s — %s", agent.get("url"), agent.get("title"))
    return agent


class CaptureResponse(BaseModel):
    file_hash: str
    url: str
    title: str
    pdf_path: str
    size: int


class ErrorResponse(BaseModel):
    detail: str


class NavigateRequest(BaseModel):
    url: str = Field(min_length=1)


class NavigateResponse(BaseModel):
    url: str
    title: str
    text: str = ""


class PelorusControlRequest(BaseModel):
    action: str
    coordinate: list[int] | None = None
    text: str | None = None
    scroll_direction: str | None = None
    scroll_amount: int | None = None
    duration: float | None = None


class BrowserReadResponse(BaseModel):
    url: str
    title: str
    text: str


async def _send_cdp_command(websocket, msg_id, method, params=None):
    """Send a CDP command and skip async event messages until its response arrives."""
    payload = {"id": msg_id, "method": method}
    if params is not None:
        payload["params"] = params

    await websocket.send(json.dumps(payload))

    while True:
        response = await websocket.recv()
        result = json.loads(response)
        if result.get("id") == msg_id:
            return result


async def get_active_tab_via_cdp():
    """Return the pinned agent tab (reused across navigations in this service)."""
    try:
        return await ensure_agent_tab()
    except Exception as e:
        logger.error(f"Failed to get active tab: {e}")
        raise


async def _dismiss_brave_content_promos(websocket) -> None:
    """Click in-page Brave consent / promo controls when they appear on a loaded tab."""
    for msg_id in (80, 81, 82):
        try:
            await _send_cdp_command(
                websocket,
                msg_id,
                "Runtime.evaluate",
                {
                    "expression": """(() => {
                      const labels = ['Disable', 'Got it', 'No thanks', 'Block', 'Not now'];
                      for (const node of document.querySelectorAll('button, [role=button], a')) {
                        const text = (node.textContent || '').trim();
                        if (labels.some((label) => text === label || text.startsWith(label))) {
                          node.click();
                          return text;
                        }
                      }
                      return '';
                    })()""",
                },
            )
        except Exception:
            break


async def _read_page_text_from_websocket(websocket) -> str:
    await _send_cdp_command(websocket, 90, "Runtime.enable")
    await _wait_for_content_ready(websocket)
    result = await _send_cdp_command(
        websocket,
        91,
        "Runtime.evaluate",
        {
            "expression": "(() => { const t = document.body ? document.body.innerText : ''; return t.slice(0, 120000); })()",
            "returnByValue": True,
        },
    )
    value = result.get("result", {}).get("result", {}).get("value")
    return str(value or "")


async def _with_active_tab_websocket():
    active_tab = await get_active_tab_via_cdp()
    websocket_url = active_tab.get("webSocketDebuggerUrl")
    if not websocket_url:
        raise HTTPException(status_code=500, detail="Could not get WebSocket URL for tab")
    return active_tab, websocket_url


async def navigate_active_tab(url: str) -> tuple[str, str, str]:
    """Navigate the pinned agent tab via CDP and return url, title, and body text."""
    validated = validate_navigation_url(url)
    _active_tab, websocket_url = await _with_active_tab_websocket()
    async with websockets.connect(websocket_url, max_size=100 * 1024 * 1024) as websocket:
        await _send_cdp_command(websocket, 1, "Page.enable")
        await _send_cdp_command(
            websocket,
            2,
            "Page.navigate",
            {"url": validated},
        )
        await _wait_for_content_ready(websocket, max_wait_ms=15000, poll_interval_ms=250)
        await _dismiss_brave_content_promos(websocket)
        title_result = await _send_cdp_command(
            websocket,
            3,
            "Runtime.evaluate",
            {"expression": "JSON.stringify({url: location.href, title: document.title})"},
        )
        tab_url = validated
        tab_title = validated
        try:
            payload = json.loads(title_result["result"]["result"]["value"])
            tab_url = str(payload.get("url") or validated)
            tab_title = str(payload.get("title") or tab_url)
        except (KeyError, TypeError, json.JSONDecodeError):
            pass
        tab_text = await _read_page_text_from_websocket(websocket)
        return tab_url, tab_title, tab_text


async def _wait_for_content_ready(websocket, max_wait_ms=5000, poll_interval_ms=200):
    """Poll CDP to detect when content is ready (adaptive wait)"""
    import time
    start_time = time.time() * 1000
    msg_id = 100
    
    while (time.time() * 1000 - start_time) < max_wait_ms:
        # Check document ready state and content length
        result = await _send_cdp_command(
            websocket,
            msg_id,
            "Runtime.evaluate",
            {
                "expression": "JSON.stringify({readyState: document.readyState, bodyLength: document.body ? document.body.innerHTML.length : 0, hasContent: document.querySelector('article, main, .content, [class*=reader], [class*=article]') !== null})"
            }
        )
        
        if 'result' in result and 'result' in result['result'] and 'value' in result['result']['result']:
            try:
                state = json.loads(result['result']['result']['value'])
                is_complete = state.get('readyState') == 'complete'
                body_length = state.get('bodyLength', 0)
                has_content_marker = state.get('hasContent', False)
                
                # Content is ready if: complete AND (has content marker OR substantial body)
                if is_complete and (has_content_marker or body_length > 2000):
                    elapsed = int(time.time() * 1000 - start_time)
                    logger.info(f"Content ready after {elapsed}ms (readyState={state.get('readyState')}, bodyLength={body_length}, hasContent={has_content_marker})")
                    return True
            except (json.JSONDecodeError, KeyError):
                pass
        
        msg_id += 1
        await asyncio.sleep(poll_interval_ms / 1000)
    
    logger.warning(f"Content ready timeout after {max_wait_ms}ms, proceeding with current state")
    return False


async def capture_page_html_via_cdp(websocket_url):
    """Capture page HTML content using CDP when PDF generation fails (optimized)"""
    logger.info(f"Capturing HTML via CDP: {websocket_url}")
    try:
        async with websockets.connect(websocket_url, max_size=100*1024*1024) as websocket:
            # Enable Page and DOM domains
            await websocket.send(json.dumps({"id": 1, "method": "Page.enable"}))
            await websocket.recv()
            await websocket.send(json.dumps({"id": 2, "method": "DOM.enable"}))
            await websocket.recv()
            
            # Adaptive wait for content (replaces fixed 5s delay)
            await _wait_for_content_ready(websocket, max_wait_ms=5000, poll_interval_ms=200)
            
            # Get document root
            await websocket.send(json.dumps({"id": 5, "method": "DOM.getDocument"}))
            response = await websocket.recv()
            result = json.loads(response)
            
            # Handle console messages that might interfere
            while 'method' in result:
                response = await websocket.recv()
                result = json.loads(response)
            
            if 'result' not in result:
                raise Exception(f"Failed to get document: {result}")
            
            root_node_id = result['result']['root']['nodeId']
            
            # Get outer HTML of root node
            await websocket.send(json.dumps({
                "id": 6,
                "method": "DOM.getOuterHTML",
                "params": {"nodeId": root_node_id}
            }))
            response = await websocket.recv()
            result = json.loads(response)
            
            # Handle console messages that might interfere
            while 'method' in result:
                response = await websocket.recv()
                result = json.loads(response)
            
            if 'result' not in result or 'outerHTML' not in result['result']:
                raise Exception(f"Failed to get outer HTML: {result}")
            
            html_content = result['result']['outerHTML']
            logger.info(f"Captured HTML content: {len(html_content)} characters")
            
            # Check if we got meaningful content (not just "Speedreader" loading)
            if len(html_content) < 1000 or ("speedreader" in html_content.lower() and len(html_content.split()) < 50):
                logger.warning("HTML content appears to be loading state, waiting longer...")
                # One additional adaptive wait
                await _wait_for_content_ready(websocket, max_wait_ms=3000, poll_interval_ms=300)
                
                # Get HTML again
                await websocket.send(json.dumps({
                    "id": 8,
                    "method": "DOM.getOuterHTML",
                    "params": {"nodeId": root_node_id}
                }))
                response = await websocket.recv()
                result = json.loads(response)
                
                # Handle console messages that might interfere
                while 'method' in result:
                    response = await websocket.recv()
                    result = json.loads(response)
                
                if 'result' in result and 'outerHTML' in result['result']:
                    html_content = result['result']['outerHTML']
                    logger.info(f"Re-captured HTML content after wait: {len(html_content)} characters")
            
            return html_content
            
    except Exception as e:
        logger.error(f"HTML capture failed: {e}")
        raise


def _sync_convert_html_to_pdf(html_content, pdf_options):
    """Synchronous WeasyPrint conversion (runs in thread pool)"""
    # Default PDF options similar to CDP
    if pdf_options is None:
        pdf_options = {
            'margin_top': '0.4in',
            'margin_bottom': '0.4in',
            'margin_left': '0.4in',
            'margin_right': '0.4in'
        }
    
    # Create HTML object
    html_doc = HTML(string=html_content)
    
    # Generate PDF
    pdf_data = html_doc.write_pdf(**pdf_options)
    
    # Validate PDF size (100MB limit)
    MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB
    if len(pdf_data) > MAX_PDF_SIZE:
        raise Exception(f"PDF size ({len(pdf_data)} bytes) exceeds maximum allowed size of {MAX_PDF_SIZE} bytes (100MB)")
    
    return pdf_data


async def convert_html_to_pdf_weasyprint(html_content, pdf_options=None):
    """Convert HTML content to PDF using WeasyPrint (async, non-blocking)"""
    try:
        # Offload CPU-bound conversion to thread pool
        loop = asyncio.get_running_loop()
        pdf_data = await loop.run_in_executor(
            _thread_pool,
            _sync_convert_html_to_pdf,
            html_content,
            pdf_options
        )
        
        logger.info(f"WeasyPrint conversion successful: {len(pdf_data)} bytes")
        return pdf_data
        
    except Exception as e:
        logger.error(f"WeasyPrint conversion failed: {e}")
        raise


def strip_pdf_link_annotations(pdf_data):
    """Remove native PDF URL link annotations while preserving page content."""
    def is_uri_link_annotation(annotation):
        if annotation.get("/Subtype") != "/Link":
            return False

        action_ref = annotation.get("/A")
        action = action_ref.get_object() if hasattr(action_ref, "get_object") else action_ref
        if action and action.get("/S") == "/URI":
            return True

        return "/URI" in annotation

    try:
        reader = PdfReader(BytesIO(pdf_data))
        writer = PdfWriter()
        removed_count = 0

        for page in reader.pages:
            annotations = page.get("/Annots")
            if annotations:
                kept_annotations = ArrayObject()
                for annotation_ref in annotations:
                    annotation = annotation_ref.get_object()
                    if is_uri_link_annotation(annotation):
                        removed_count += 1
                        continue
                    kept_annotations.append(annotation_ref)

                if kept_annotations:
                    page[NameObject("/Annots")] = kept_annotations
                else:
                    page.pop(NameObject("/Annots"), None)

            writer.add_page(page)

        if removed_count == 0:
            return pdf_data

        output = BytesIO()
        writer.write(output)
        sanitized_pdf = output.getvalue()
        logger.info(f"Removed {removed_count} native PDF URL link annotations from capture")
        return sanitized_pdf
    except Exception as e:
        logger.warning(f"Failed to strip PDF link annotations; using original PDF: {e}")
        return pdf_data


async def prepare_page_for_pdf(websocket, max_wait_ms=CAPTURE_READY_TIMEOUT_MS):
    """Best-effort page settling before printToPDF, bounded by max_wait_ms."""
    script = f"""
    (async () => {{
      const deadline = Date.now() + {max_wait_ms};
      const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
      const remaining = () => Math.max(0, deadline - Date.now());
      const timed = (promise, ms) => Promise.race([promise, sleep(Math.max(0, ms))]);

      const state = {{
        readyState: document.readyState,
        bodyLength: document.body ? document.body.innerHTML.length : 0,
        imageCount: document.images ? document.images.length : 0,
        pendingImages: 0,
        scrolled: false,
        timedOut: false,
      }};

      while (document.readyState !== "complete" && remaining() > 0) {{
        await sleep(100);
      }}

      try {{
        if (document.fonts && document.fonts.ready && remaining() > 0) {{
          await timed(document.fonts.ready, Math.min(1500, remaining()));
        }}
      }} catch (_err) {{}}

      const originalX = window.scrollX || 0;
      const originalY = window.scrollY || 0;
      const doc = document.documentElement;
      const body = document.body;
      const scrollHeight = Math.max(
        body ? body.scrollHeight : 0,
        doc ? doc.scrollHeight : 0
      );
      const viewportHeight = window.innerHeight || 800;
      const steps = Math.min(10, Math.max(1, Math.ceil(scrollHeight / viewportHeight)));

      if (scrollHeight > viewportHeight * 1.25) {{
        for (let i = 0; i < steps && remaining() > 250; i += 1) {{
          const y = Math.round((scrollHeight - viewportHeight) * (i / Math.max(1, steps - 1)));
          window.scrollTo(0, y);
          await sleep(150);
        }}
        window.scrollTo(originalX, originalY);
        await sleep(Math.min(250, remaining()));
        state.scrolled = true;
      }}

      while (remaining() > 0) {{
        const images = Array.from(document.images || []);
        const pending = images.filter((img) => !img.complete).length;
        state.pendingImages = pending;
        if (pending === 0) break;
        await sleep(200);
      }}

      state.readyState = document.readyState;
      state.bodyLength = document.body ? document.body.innerHTML.length : 0;
      state.pendingImages = Array.from(document.images || []).filter((img) => !img.complete).length;
      state.timedOut = remaining() <= 0;
      return JSON.stringify(state);
    }})()
    """

    try:
        result = await _send_cdp_command(
            websocket,
            50,
            "Runtime.evaluate",
            {
                "expression": script,
                "awaitPromise": True,
                "returnByValue": True,
            },
        )
        value = result.get("result", {}).get("result", {}).get("value")
        state = json.loads(value) if value else {}
        if state.get("timedOut"):
            logger.warning(f"Page readiness cutoff hit after {max_wait_ms}ms: {state}")
        else:
            logger.info(f"Page prepared for PDF: {state}")
        return state
    except Exception as e:
        logger.warning(f"Page readiness preparation failed; continuing to PDF: {e}")
        return {}


async def generate_pdf_via_cdp(websocket_url, pdf_options):
    """Generate PDF using direct CDP WebSocket connection"""
    logger.info(f"Connecting to WebSocket: {websocket_url}")
    try:
        # Increase max_size to 100MB to handle large PDFs
        async with websockets.connect(websocket_url, max_size=100*1024*1024) as websocket:
            # Enable Page domain
            await _send_cdp_command(websocket, 1, "Page.enable")

            await prepare_page_for_pdf(websocket)
            
            # Generate PDF
            result = await _send_cdp_command(websocket, 2, "Page.printToPDF", pdf_options)

            if 'result' in result and 'data' in result['result']:
                pdf_data = base64.b64decode(result['result']['data'])
                # Validate PDF size (100MB limit)
                MAX_PDF_SIZE = 100 * 1024 * 1024  # 100MB
                if len(pdf_data) > MAX_PDF_SIZE:
                    raise Exception(f"PDF size ({len(pdf_data)} bytes) exceeds maximum allowed size of {MAX_PDF_SIZE} bytes (100MB)")
                return pdf_data
            else:
                raise Exception(f"PDF generation failed: {result}")
                
    except Exception as e:
        logger.error(f"CDP PDF generation failed: {e}")
        raise


@app.get("/health")
async def health():
    """Health check for direct CDP service (optimized with pooled session)"""
    try:
        # Check if browser debug interface is accessible and get tabs in one flow
        tabs_response = _cdp_session.get(f"http://{BROWSER_DEBUG_URL}/json", timeout=5)
        if tabs_response.status_code != 200:
            return {"status": "unhealthy", "browser": "debug_interface_not_accessible"}
        
        tabs = tabs_response.json()
        if not tabs:
            return {"status": "unhealthy", "browser": "no_tabs_available"}
        
        # Get browser version info (reusing pooled connection)
        version_response = _cdp_session.get(f"http://{BROWSER_DEBUG_URL}/json/version", timeout=5)
        browser_version = "unknown"
        if version_response.status_code == 200:
            browser_info = version_response.json()
            browser_version = browser_info.get('Browser', 'unknown')
        
        # Return healthy status with browser info
        pelorus_ok = await pelorus_ready()
        pelorus_env = str(os.environ.get("PELORUS") or "").lower() in {"1", "true", "yes"}
        if pelorus_env and pelorus_ok:
            control_backend = "pelorus"
        elif cdp_debug_reachable():
            control_backend = "cdp"
        else:
            control_backend = "unavailable"
        return {
            "status": "healthy", 
            "browser": "cdp_accessible", 
            "version": f"direct_cdp_{browser_version}",
            "tabs_count": len(tabs),
            "method": "direct_cdp_websocket",
            "pelorus": "ready" if pelorus_ok else "unavailable",
            "desktop_control": pelorus_ok,
            "control_backend": control_backend,
            **(
                {"desktop_control_note": CDP_FALLBACK_NOTE}
                if control_backend == "cdp"
                else (
                    {}
                    if pelorus_ok
                    else {"desktop_control_reason": DESKTOP_CONTROL_UNAVAILABLE_MESSAGE}
                )
            ),
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "browser": str(e)}


@app.post("/navigate", response_model=NavigateResponse)
async def navigate_page(body: NavigateRequest):
    try:
        url, title, text = await navigate_active_tab(body.url)
        return NavigateResponse(url=url, title=title, text=text)
    except BrowserUrlPolicyError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": exc.message},
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Navigate failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Navigate failed: {exc}") from exc


@app.get("/pelorus/state")
async def pelorus_state():
    if not await pelorus_ready():
        return await pelorus_get("/api/state")
    try:
        return await pelorus_get("/api/state")
    except Exception as exc:
        logger.error("Pelorus state failed: %s", exc)
        return {
            "desktop_control": False,
            "code": "desktop_control_unavailable",
            "message": str(exc),
        }


@app.get("/pelorus/screenshot")
async def pelorus_screenshot():
    if not await pelorus_ready():
        return {
            "desktop_control": False,
            "code": "desktop_control_unavailable",
            "message": DESKTOP_CONTROL_UNAVAILABLE_MESSAGE,
            "image": None,
        }
    try:
        payload = await desktop_screenshot()
        if isinstance(payload, dict):
            payload.setdefault("desktop_control", True)
        return payload
    except Exception as exc:
        logger.error("Pelorus screenshot failed: %s", exc)
        return {
            "desktop_control": False,
            "code": "desktop_control_unavailable",
            "message": str(exc),
            "image": None,
        }


@app.post("/pelorus/control")
async def pelorus_control(body: PelorusControlRequest):
    if not await pelorus_ready():
        return {
            "desktop_control": False,
            "code": "desktop_control_unavailable",
            "message": DESKTOP_CONTROL_UNAVAILABLE_MESSAGE,
        }
    payload: dict[str, Any] = {"action": body.action}
    if body.coordinate is not None:
        payload["coordinate"] = body.coordinate
    if body.text is not None:
        payload["text"] = body.text
    if body.scroll_direction is not None:
        payload["scroll_direction"] = body.scroll_direction
    if body.scroll_amount is not None:
        payload["scroll_amount"] = body.scroll_amount
    if body.duration is not None:
        payload["duration"] = body.duration
    try:
        return await pelorus_post("/api/desktop/control", payload)
    except Exception as exc:
        logger.error("Pelorus control failed: %s", exc)
        raise HTTPException(status_code=503, detail={"code": "pelorus_control_failed", "message": str(exc)}) from exc


@app.get("/read", response_model=BrowserReadResponse)
async def read_active_page():
    try:
        active_tab, websocket_url = await _with_active_tab_websocket()
        tab_url = str(active_tab.get("url") or "")
        tab_title = str(active_tab.get("title") or tab_url)
        async with websockets.connect(websocket_url, max_size=100 * 1024 * 1024) as websocket:
            await _send_cdp_command(websocket, 1, "Runtime.enable")
            await _wait_for_content_ready(websocket)
            await _dismiss_brave_content_promos(websocket)
            result = await _send_cdp_command(
                websocket,
                2,
                "Runtime.evaluate",
                {
                    "expression": "(() => { const t = document.body ? document.body.innerText : ''; return JSON.stringify({url: location.href, title: document.title, text: t.slice(0, 120000)}); })()",
                    "returnByValue": True,
                },
            )
            raw = result.get("result", {}).get("result", {}).get("value", "{}")
            payload = json.loads(raw) if isinstance(raw, str) else {}
            return BrowserReadResponse(
                url=str(payload.get("url") or tab_url),
                title=str(payload.get("title") or tab_title),
                text=str(payload.get("text") or ""),
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Read page failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Read failed: {exc}") from exc


@app.post("/capture", response_model=CaptureResponse)
async def capture_page():
    """
    Capture the current browser page as PDF using direct CDP WebSocket.
    Connects to existing browser, finds active tab, prints to PDF, saves to /captures.
    """
    try:
        logger.info(f"Connecting to browser at {BROWSER_DEBUG_URL}")
        
        # Get active tab
        active_tab = await get_active_tab_via_cdp()
        websocket_url = active_tab.get('webSocketDebuggerUrl')
        logger.info(f"WebSocket URL: {websocket_url}")
        
        if not websocket_url:
            raise HTTPException(status_code=500, detail="Could not get WebSocket URL for tab")
        
        tab_url = active_tab.get('url', '')
        tab_title = active_tab.get('title', '')
        if tab_url.casefold().startswith("about:"):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "browser_capture_blank_page",
                    "message": "Browser capture cannot index a blank or internal about: page. Open a web page first.",
                },
            )
        if not tab_url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "browser_capture_invalid_url",
                    "message": "Browser capture did not return a capturable http(s) page. Open a web page before capturing.",
                },
            )

        # Generate PDF using CDP - use numeric values (inches)
        pdf_options = {
            "landscape": False,
            "displayHeaderFooter": False,
            "printBackground": True,
            "preferCSSPageSize": True,
            "marginTop": 0.4,
            "marginBottom": 0.4,
            "marginLeft": 0.4,
            "marginRight": 0.4
        }
        
        try:
            pdf_data = await generate_pdf_via_cdp(websocket_url, pdf_options)
            logger.info("CDP PDF generation successful")
        except Exception as cdp_error:
            logger.warning(f"CDP PDF generation failed: {cdp_error}")
            # Check if it's the "Printing is not available" error
            if "Printing is not available" in str(cdp_error):
                logger.info("Falling back to HTML capture + WeasyPrint conversion")
                try:
                    # Capture HTML content
                    html_content = await capture_page_html_via_cdp(websocket_url)
                    
                    # Convert HTML to PDF using WeasyPrint
                    pdf_data = await convert_html_to_pdf_weasyprint(html_content)
                    logger.info("WeasyPrint fallback successful")
                except Exception as fallback_error:
                    logger.error(f"WeasyPrint fallback also failed: {fallback_error}")
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Both CDP and WeasyPrint PDF generation failed. CDP error: {cdp_error}. WeasyPrint error: {fallback_error}"
                    )
            else:
                # Re-raise the original CDP error if it's not the "Printing is not available" error
                raise
        
        pdf_data = strip_pdf_link_annotations(pdf_data)

        # Compute hash and save directly to final path (streaming, no temp file)
        file_hash = hashlib.md5(pdf_data).hexdigest()
        final_path = CAPTURES_DIR / f"{file_hash}.pdf"
        
        with open(final_path, 'wb') as f:
            f.write(pdf_data)
        
        logger.info(f"Capture complete: {file_hash} ({len(pdf_data)} bytes)")
        
        return CaptureResponse(
            file_hash=file_hash,
            url=tab_url,
            title=tab_title or tab_url,
            pdf_path=str(final_path),
            size=len(pdf_data)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        raise HTTPException(status_code=500, detail=f"Capture failed: {str(e)}")


@app.get("/captures/{file_hash}")
async def get_capture(file_hash: str):
    """Download a captured PDF by hash."""
    if not FILE_HASH_PATTERN.fullmatch(file_hash):
        raise HTTPException(status_code=400, detail="Invalid file hash")

    pdf_path = CAPTURES_DIR / f"{file_hash}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"capture-{file_hash}.pdf"
    )


@app.get("/captures/{file_hash}/exists")
async def check_capture_exists(file_hash: str):
    """Check if a capture exists without downloading."""
    if not FILE_HASH_PATTERN.fullmatch(file_hash):
        raise HTTPException(status_code=400, detail="Invalid file hash")

    pdf_path = CAPTURES_DIR / f"{file_hash}.pdf"
    return {"exists": pdf_path.exists()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
