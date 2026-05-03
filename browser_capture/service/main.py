"""
Browser Capture API - Simplified Playwright Implementation
Clean service that captures browser pages as PDF using Playwright.
"""

import os
import hashlib
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Browser Capture Service - Playwright")

CAPTURES_DIR = Path("/captures")
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

BROWSER_DEBUG_URL = os.environ.get("BROWSER_DEBUG_URL", "localhost:9222")


class CaptureResponse(BaseModel):
    file_hash: str
    url: str
    title: str
    pdf_path: str
    size: int


class ErrorResponse(BaseModel):
    detail: str


async def get_browser_context():
    """Connect to existing browser using CDP and return Playwright context"""
    try:
        # Import Playwright at the top level to avoid import issues
        import playwright.async_api
        
        # Start Playwright
        playwright = await playwright.async_api.async_playwright.start()
        
        # Connect to existing Chrome instance via CDP
        browser = await playwright.chromium.connect_over_cdp(f"ws://{BROWSER_DEBUG_URL}")
        
        # Get existing pages/tabs
        pages = browser.contexts[0].pages if browser.contexts else []
        
        # Find the most recently active page
        active_page = None
        for page in pages:
            # Skip internal pages
            if page.url.startswith(('chrome://', 'chrome-extension://', 'about:')):
                continue
            active_page = page
            break
        
        if not active_page and pages:
            # Fallback to first non-system page
            for page in pages:
                if not page.url.startswith(('chrome://', 'chrome-extension://', 'about:')):
                    active_page = page
                    break
        
        if not active_page:
            # Last resort - first page
            active_page = pages[0] if pages else None
        
        if not active_page:
            raise HTTPException(status_code=500, detail="No suitable browser page found")
        
        logger.info(f"Selected page: {active_page.url} - {await active_page.title()}")
        return active_page
        
    except Exception as e:
        logger.error(f"Failed to connect to browser: {e}")
        raise HTTPException(status_code=500, detail=f"Browser connection failed: {str(e)}")


async def capture_page_as_pdf(page):
    """Capture page as PDF using Playwright with fallback for reader mode"""
    try:
        logger.info(f"Capturing PDF for page: {page.url}")
        
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
            
            # Fallback: Get HTML content and convert to PDF
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
            
            # Convert HTML to PDF using Playwright's page.pdf() with HTML content
            # This is a workaround - we'll create a new page with the HTML content
            temp_context = await page.context.browser.new_context()
            temp_page = await temp_context.new_page()
            
            try:
                await temp_page.set_content(html_content, wait_until='networkidle')
                await temp_page.wait_for_timeout(2000)
                
                pdf_data = await temp_page.pdf(**pdf_options)
                logger.info(f"HTML-to-PDF conversion successful: {len(pdf_data)} bytes")
                return pdf_data
                
            finally:
                await temp_context.close()
                
    except Exception as e:
        logger.error(f"PDF capture failed: {e}")
        raise HTTPException(status_code=500, detail=f"PDF capture failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check for Playwright service"""
    try:
        # Try to connect to browser
        page = await get_browser_context()
        if page:
            return {
                "status": "healthy", 
                "browser": "playwright_connected",
                "method": "playwright_cdp",
                "page_url": page.url
            }
    except Exception as e:
        return {"status": "unhealthy", "browser": str(e)}


@app.post("/capture", response_model=CaptureResponse)
async def capture_page():
    """
    Capture the current browser page as PDF using Playwright.
    Connects to existing browser, finds active tab, converts to PDF, saves to /captures.
    """
    try:
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
            title=await page.title(),
            pdf_path=str(final_path),
            size=len(pdf_data)
        )
        
    except Exception as e:
        logger.error(f"Capture failed: {e}")
        raise HTTPException(status_code=500, detail=f"Capture failed: {str(e)}")


@app.get("/captures/{file_hash}")
async def get_capture(file_hash: str):
    """Download a captured PDF by hash."""
    import re
    FILE_HASH_PATTERN = re.compile(r"^[a-f0-9]{32}$")
    
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
    import re
    FILE_HASH_PATTERN = re.compile(r"^[a-f0-9]{32}$")
    
    if not FILE_HASH_PATTERN.fullmatch(file_hash):
        raise HTTPException(status_code=400, detail="Invalid file hash")

    pdf_path = CAPTURES_DIR / f"{file_hash}.pdf"
    return {"exists": pdf_path.exists()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
