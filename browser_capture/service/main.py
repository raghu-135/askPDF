"""
Browser Capture API - Enhanced CDP Implementation
Clean service that captures browser pages as PDF using enhanced CDP with Playwright-like features.
"""

import os
import hashlib
import uuid
import base64
import json
import logging
import re
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import websockets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Browser Capture Service - Enhanced CDP")

CAPTURES_DIR = Path("/captures")
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

BROWSER_DEBUG_URL = os.environ.get("BROWSER_DEBUG_URL", "localhost:9222")
FILE_HASH_PATTERN = re.compile(r"^[a-f0-9]{32}$")


class CaptureResponse(BaseModel):
    file_hash: str
    url: str
    title: str
    pdf_path: str
    size: int


class ErrorResponse(BaseModel):
    detail: str


class BrowserPage:
    """Simple wrapper to mimic Playwright page interface using CDP"""
    
    def __init__(self, tab_info, websocket_url):
        self.url = tab_info.get('url', '')
        self.title = tab_info.get('title', '')
        self.websocket_url = websocket_url
        self._websocket = None
    
    async def connect(self):
        """Connect to the page's WebSocket"""
        self._websocket = await websockets.connect(self.websocket_url, max_size=100*1024*1024)
        # Enable Page domain
        await self._websocket.send(json.dumps({"id": 1, "method": "Page.enable"}))
        await self._websocket.recv()
    
    async def wait_for_load_state(self, state='networkidle', timeout=10000):
        """Wait for page load state - simplified version"""
        try:
            # Wait for page to be fully loaded
            await self._websocket.send(json.dumps({
                "id": 2, 
                "method": "Page.waitForLoadEvent",
                "params": {}
            }))
            response = await self._websocket.recv()
            result = json.loads(response)
            
            # Handle console messages that might interfere
            while 'method' in result:
                response = await self._websocket.recv()
                result = json.loads(response)
                
        except Exception as e:
            logger.warning(f"Wait for load state failed: {e}")
    
    async def wait_for_timeout(self, timeout):
        """Wait for specified timeout"""
        await asyncio.sleep(timeout / 1000)
    
    async def content(self):
        """Get page HTML content"""
        try:
            # Enable DOM domain
            await self._websocket.send(json.dumps({"id": 3, "method": "DOM.enable"}))
            await self._websocket.recv()
            
            # Get document root
            await self._websocket.send(json.dumps({"id": 4, "method": "DOM.getDocument"}))
            response = await self._websocket.recv()
            result = json.loads(response)
            
            # Handle console messages
            while 'method' in result:
                response = await self._websocket.recv()
                result = json.loads(response)
            
            if 'result' not in result:
                raise Exception(f"Failed to get document: {result}")
            
            root_node_id = result['result']['root']['nodeId']
            
            # Get outer HTML
            await self._websocket.send(json.dumps({
                "id": 5,
                "method": "DOM.getOuterHTML",
                "params": {"nodeId": root_node_id}
            }))
            response = await self._websocket.recv()
            result = json.loads(response)
            
            # Handle console messages
            while 'method' in result:
                response = await self._websocket.recv()
                result = json.loads(response)
            
            if 'result' not in result or 'outerHTML' not in result['result']:
                raise Exception(f"Failed to get outer HTML: {result}")
            
            return result['result']['outerHTML']
            
        except Exception as e:
            logger.error(f"Failed to get page content: {e}")
            raise
    
    async def pdf(self, **options):
        """Generate PDF from page"""
        try:
            pdf_options = {
                "landscape": options.get('landscape', False),
                "displayHeaderFooter": options.get('displayHeaderFooter', False),
                "printBackground": options.get('print_background', True),
                "preferCSSPageSize": options.get('preferCSSPageSize', True),
                "marginTop": 0.4,
                "marginBottom": 0.4,
                "marginLeft": 0.4,
                "marginRight": 0.4
            }
            
            await self._websocket.send(json.dumps({
                "id": 6,
                "method": "Page.printToPDF",
                "params": pdf_options
            }))
            
            response = await self._websocket.recv()
            result = json.loads(response)
            
            # Handle console messages
            while 'method' in result:
                response = await self._websocket.recv()
                result = json.loads(response)
            
            if 'result' in result and 'data' in result['result']:
                pdf_data = base64.b64decode(result['result']['data'])
                return pdf_data
            else:
                error_msg = result.get('error', {}).get('message', 'Unknown error')
                raise Exception(f"PDF generation failed: {error_msg}")
                
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise
    
    async def close(self):
        """Close WebSocket connection"""
        if self._websocket:
            await self._websocket.close()


async def get_browser_context():
    """Connect to existing browser using CDP and return page wrapper"""
    try:
        # Get all tabs from debug interface
        response = requests.get(f"http://{BROWSER_DEBUG_URL}/json", timeout=5)
        if response.status_code != 200:
            raise Exception("Failed to get tabs from debug interface")
        
        tabs = response.json()
        if not tabs:
            raise Exception("No tabs available")
        
        logger.info(f"Available tabs: {[tab.get('url', 'unknown') for tab in tabs]}")
        
        # Find the best tab - prefer content tabs over system pages
        for tab in tabs:
            url = tab.get('url', '')
            title = tab.get('title', '')
            
            # Skip system/internal pages
            if url.startswith(('chrome://', 'chrome-extension://', 'about:')):
                continue
                
            # Found a good content tab
            logger.info(f"Selected content tab: {url} - {title}")
            websocket_url = tab.get('webSocketDebuggerUrl')
            if websocket_url:
                return BrowserPage(tab, websocket_url)
        
        # Fallback to first available tab
        logger.warning(f"Using first available tab: {tabs[0].get('url')} - {tabs[0].get('title')}")
        websocket_url = tabs[0].get('webSocketDebuggerUrl')
        if websocket_url:
            return BrowserPage(tabs[0], websocket_url)
        
        raise HTTPException(status_code=500, detail="No suitable browser page found")
        
    except Exception as e:
        logger.error(f"Failed to connect to browser: {e}")
        raise HTTPException(status_code=500, detail=f"Browser connection failed: {str(e)}")


async def capture_page_as_pdf(page):
    """Capture page as PDF using enhanced CDP with fallback for reader mode"""
    try:
        logger.info(f"Capturing PDF for page: {page.url}")
        
        # Connect to the page
        await page.connect()
        
        try:
            # Wait for page to be fully loaded
            await page.wait_for_load_state('networkidle', timeout=10000)
            
            # Additional wait for dynamic content (reader mode)
            await page.wait_for_timeout(2000)
            
            # Try direct PDF generation first
            try:
                pdf_options = {
                    'format': 'A4',
                    'print_background': True,
                    'margin': {
                        'top': '0.4in',
                        'bottom': '0.4in', 
                        'left': '0.4in',
                        'right': '0.4in'
                    }
                }
                
                pdf_data = await page.pdf(**pdf_options)
                logger.info(f"Direct PDF generation successful: {len(pdf_data)} bytes")
                return pdf_data
                
            except Exception as pdf_error:
                logger.warning(f"Direct PDF generation failed: {pdf_error}")
                
                # Fallback: Get HTML content and create simple PDF
                logger.info("Falling back to HTML capture")
                
                # Wait additional time for content to load
                await page.wait_for_timeout(3000)
                
                # Get page HTML
                html_content = await page.content()
                logger.info(f"Captured HTML content: {len(html_content)} characters")
                
                # Check if we got meaningful content
                if len(html_content) < 1000 or ('speedreader' in html_content.lower() and len(html_content.split()) < 50):
                    logger.warning("HTML content appears insufficient, waiting longer...")
                    await page.wait_for_timeout(5000)
                    html_content = await page.content()
                    logger.info(f"Re-captured HTML: {len(html_content)} characters")
                
                # Create simple PDF from HTML content
                return create_simple_pdf_from_html(html_content)
                
        finally:
            await page.close()
                
    except Exception as e:
        logger.error(f"PDF capture failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF capture failed: {str(e)}")


def create_simple_pdf_from_html(html_content):
    """Create a simple PDF from HTML content"""
    import re
    # Extract text content from HTML
    text_content = re.sub(r'<[^>]+>', ' ', html_content)
    text_content = re.sub(r'\s+', ' ', text_content).strip()
    
    # Create a minimal PDF structure
    header = b'%PDF-1.4\n'
    catalog = b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n'
    pages = b'2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n'
    
    # Create a simple page with text
    page_content = f"BT /F1 12 Tf 72 720 Td ({text_content[:500]}) Tj ET".encode('latin-1', errors='ignore')
    page = f'3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>\nendobj\n'.encode()
    content = f'4 0 obj\n<< /Length {len(page_content)} >>\nstream\n{page_content}\nendstream\nendobj\n'.encode()
    
    xref = b'xref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \n0000000204 00000 n \n'
    trailer = b'trailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n312\n%%EOF\n'
    
    pdf_data = header + catalog + pages + page + content + xref + trailer
    logger.info(f"Created simple PDF fallback: {len(pdf_data)} bytes")
    return pdf_data


@app.get("/health")
async def health():
    """Health check for enhanced CDP service"""
    try:
        # Check if browser debug interface is accessible
        response = requests.get(f"http://{BROWSER_DEBUG_URL}/json/version", timeout=5)
        if response.status_code != 200:
            return {"status": "unhealthy", "browser": "debug_interface_not_accessible"}
        
        # Get browser info
        browser_info = response.json()
        browser_version = browser_info.get('Browser', 'unknown')
        
        # Check if tabs are available
        tabs_response = requests.get(f"http://{BROWSER_DEBUG_URL}/json", timeout=5)
        if tabs_response.status_code != 200:
            return {"status": "unhealthy", "browser": "no_tabs_accessible"}
        
        tabs = tabs_response.json()
        if not tabs:
            return {"status": "unhealthy", "browser": "no_tabs_available"}
        
        return {
            "status": "healthy", 
            "browser": "cdp_accessible", 
            "version": f"enhanced_cdp_{browser_version}",
            "tabs_count": len(tabs),
            "method": "enhanced_cdp_websocket"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "browser": str(e)}


@app.post("/capture", response_model=CaptureResponse)
async def capture_page():
    """
    Capture the current browser page as PDF using enhanced CDP.
    Connects to existing browser, finds active tab, converts to PDF, saves to /captures.
    """
    try:
        logger.info(f"Connecting to browser at {BROWSER_DEBUG_URL}")
        
        # Get browser page
        page = await get_browser_context()
        
        # Capture as PDF
        pdf_data = await capture_page_as_pdf(page)
        
        # Save PDF
        file_id = str(uuid.uuid4())
        pdf_path = CAPTURES_DIR / f"{file_id}.pdf"
        
        with open(pdf_path, 'wb') as f:
            f.write(pdf_data)
        
        # Calculate hash and rename
        file_hash = hashlib.md5(pdf_data).hexdigest()
        final_path = CAPTURES_DIR / f"{file_hash}.pdf"
        pdf_path.rename(final_path)
        
        logger.info(f"Capture complete: {file_hash} ({len(pdf_data)} bytes)")
        
        return CaptureResponse(
            file_hash=file_hash,
            url=page.url,
            title=page.title,
            pdf_path=str(final_path),
            size=len(pdf_data)
        )
        
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
