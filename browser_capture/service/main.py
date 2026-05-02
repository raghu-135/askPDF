"""
Browser Capture API
Simple FastAPI service that captures the current browser page as PDF using Playwright.
"""

import os
import hashlib
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Browser Capture Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CAPTURES_DIR = Path("/captures")
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

CDP_URL = os.environ.get("CDP_URL", "http://localhost:9222")


class CaptureResponse(BaseModel):
    file_hash: str
    url: str
    title: str
    pdf_path: str
    size: int


class ErrorResponse(BaseModel):
    detail: str


@app.get("/health")
async def health():
    """Health check - also verifies browser is accessible."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(CDP_URL, timeout=5000)
            version = browser.version  # property, not coroutine
            await browser.close()
        return {"status": "healthy", "browser": "connected", "version": version}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "browser": str(e)}


@app.post("/capture", response_model=CaptureResponse)
async def capture_page():
    """
    Capture the current browser page as PDF.
    Connects via CDP, prints to PDF, saves to /captures.
    """
    try:
        logger.info(f"Connecting to browser at {CDP_URL}")
        
        async with async_playwright() as p:
            browser = await p.chromium.connect_over_cdp(CDP_URL)
            
            # Get first context and page
            context = browser.contexts[0] if browser.contexts else await browser.new_context()
            page = context.pages[0] if context.pages else await context.new_page()
            
            # Get page info
            url = page.url
            title = await page.title()
            
            logger.info(f"Capturing page: {url} (title: {title})")
            
            # Generate filename
            file_id = str(uuid.uuid4())
            pdf_path = CAPTURES_DIR / f"{file_id}.pdf"
            
            # Print to PDF
            await page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={"top": "20px", "right": "20px", "bottom": "20px", "left": "20px"}
            )
            
            logger.info(f"PDF saved to {pdf_path}")
            
            # Calculate hash
            pdf_bytes = pdf_path.read_bytes()
            file_hash = hashlib.md5(pdf_bytes).hexdigest()
            
            # Rename to hash-based filename
            final_path = CAPTURES_DIR / f"{file_hash}.pdf"
            pdf_path.rename(final_path)
            
            await browser.close()
            
            logger.info(f"Capture complete: {file_hash} ({len(pdf_bytes)} bytes)")
            
            return CaptureResponse(
                file_hash=file_hash,
                url=url,
                title=title or url,
                pdf_path=str(final_path),
                size=len(pdf_bytes)
            )
            
    except Exception as e:
        logger.error(f"Capture failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Capture failed: {str(e)}")


@app.get("/captures/{file_hash}")
async def get_capture(file_hash: str):
    """Download a captured PDF by hash."""
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
    pdf_path = CAPTURES_DIR / f"{file_hash}.pdf"
    return {"exists": pdf_path.exists()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
