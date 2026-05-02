"""
Browser Capture API - Direct CDP Implementation
Simple FastAPI service that captures the current browser page as PDF using direct Chrome DevTools Protocol.
"""

import os
import hashlib
import uuid
import base64
import json
import logging
import re
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import requests
import websockets
from weasyprint import HTML, CSS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Browser Capture Service - Direct CDP")

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


async def get_active_tab_via_cdp():
    """Get the currently active tab using direct CDP calls"""
    try:
        # Get all tabs from debug interface
        response = requests.get(f"http://{BROWSER_DEBUG_URL}/json", timeout=5)
        if response.status_code != 200:
            raise Exception("Failed to get tabs from debug interface")
        
        tabs = response.json()
        if not tabs:
            raise Exception("No tabs available")
        
        # Log all available tabs for debugging
        logger.info(f"Available tabs: {[tab.get('url', 'unknown') for tab in tabs]}")
        
        # Find the best tab to capture - prefer content tabs over system panels
        content_tab = None
        speedreader_tab = None
        
        for tab in tabs:
            url = tab.get('url', '')
            title = tab.get('title', '')
            
            # Skip system/internal pages
            if url.startswith(('chrome://', 'chrome-extension://', 'about:')):
                continue
                
            # Look for reader mode content
            if 'speedreader' in title.lower() or 'reader' in title.lower():
                if url.startswith('http'):
                    content_tab = tab
                    break
                    
            # Look for regular web content as fallback
            if url.startswith(('http://', 'https://')):
                if not content_tab:  # First content tab
                    content_tab = tab
        
        # If we found a content tab, use it
        if content_tab:
            logger.info(f"Selected content tab: {content_tab.get('url')} - {content_tab.get('title')}")
            return content_tab
        
        # Fallback to first non-system tab
        for tab in tabs:
            url = tab.get('url', '')
            if not url.startswith(('chrome://', 'chrome-extension://', 'about:')):
                logger.info(f"Fallback to tab: {url} - {tab.get('title')}")
                return tab
        
        # Last resort - first tab
        logger.warning(f"Using first available tab: {tabs[0].get('url')} - {tabs[0].get('title')}")
        return tabs[0]
        
    except Exception as e:
        logger.error(f"Failed to get active tab: {e}")
        raise


async def capture_page_html_via_cdp(websocket_url):
    """Capture page HTML content using CDP when PDF generation fails"""
    logger.info(f"Capturing HTML via CDP: {websocket_url}")
    try:
        async with websockets.connect(websocket_url, max_size=100*1024*1024) as websocket:
            # Enable Page and DOM domains
            await websocket.send(json.dumps({"id": 1, "method": "Page.enable"}))
            await websocket.recv()
            await websocket.send(json.dumps({"id": 2, "method": "DOM.enable"}))
            await websocket.recv()
            
            # Wait for page to be fully loaded, especially for reader mode content
            await websocket.send(json.dumps({
                "id": 3,
                "method": "Page.waitForLoadEvent",
                "params": {}
            }))
            response = await websocket.recv()
            result = json.loads(response)
            
            # Handle console messages that might interfere
            while 'method' in result:
                response = await websocket.recv()
                result = json.loads(response)
            
            # Additional wait for dynamic content (reader mode)
            await websocket.send(json.dumps({
                "id": 4,
                "method": "Runtime.evaluate",
                "params": {
                    "expression": "new Promise(resolve => setTimeout(resolve, 2000))"
                }
            }))
            response = await websocket.recv()
            result = json.loads(response)
            
            # Handle console messages that might interfere
            while 'method' in result:
                response = await websocket.recv()
                result = json.loads(response)
            
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
            if len(html_content) < 1000 or "speedreader" in html_content.lower() and len(html_content.split()) < 50:
                logger.warning("HTML content appears to be loading state, waiting longer...")
                # Wait additional time and try once more
                await websocket.send(json.dumps({
                    "id": 7,
                    "method": "Runtime.evaluate",
                    "params": {
                        "expression": "new Promise(resolve => setTimeout(resolve, 3000))"
                    }
                }))
                response = await websocket.recv()
                result = json.loads(response)
                
                # Handle console messages that might interfere
                while 'method' in result:
                    response = await websocket.recv()
                    result = json.loads(response)
                
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


def convert_html_to_pdf_weasyprint(html_content, pdf_options=None):
    """Convert HTML content to PDF using WeasyPrint"""
    try:
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
        
        logger.info(f"WeasyPrint conversion successful: {len(pdf_data)} bytes")
        return pdf_data
        
    except Exception as e:
        logger.error(f"WeasyPrint conversion failed: {e}")
        raise


async def generate_pdf_via_cdp(websocket_url, pdf_options):
    """Generate PDF using direct CDP WebSocket connection"""
    logger.info(f"Connecting to WebSocket: {websocket_url}")
    try:
        # Increase max_size to 100MB to handle large PDFs
        async with websockets.connect(websocket_url, max_size=100*1024*1024) as websocket:
            # Enable Page domain
            await websocket.send(json.dumps({
                "id": 1,
                "method": "Page.enable"
            }))
            await websocket.recv()  # Acknowledgment
            
            # Generate PDF
            await websocket.send(json.dumps({
                "id": 2,
                "method": "Page.printToPDF",
                "params": pdf_options
            }))
            
            response = await websocket.recv()
            result = json.loads(response)

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
    """Health check for direct CDP service"""
    try:
        # Check if browser debug interface is accessible
        response = requests.get(f"http://{BROWSER_DEBUG_URL}/json/version", timeout=5)
        if response.status_code != 200:
            return {"status": "unhealthy", "browser": "debug_interface_not_accessible"}
        
        # Get browser info from debug interface
        browser_info = response.json()
        browser_version = browser_info.get('Browser', 'unknown')
        
        # Check if tabs are available
        tabs_response = requests.get(f"http://{BROWSER_DEBUG_URL}/json", timeout=5)
        if tabs_response.status_code != 200:
            return {"status": "unhealthy", "browser": "no_tabs_accessible"}
        
        tabs = tabs_response.json()
        if not tabs:
            return {"status": "unhealthy", "browser": "no_tabs_available"}
        
        # Return healthy status with browser info
        return {
            "status": "healthy", 
            "browser": "cdp_accessible", 
            "version": f"direct_cdp_{browser_version}",
            "tabs_count": len(tabs),
            "method": "direct_cdp_websocket"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "browser": str(e)}


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
                    pdf_data = convert_html_to_pdf_weasyprint(html_content)
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
            url=tab_url,
            title=tab_title or tab_url,
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
