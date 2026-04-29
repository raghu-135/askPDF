"""
main.py - FastAPI entrypoint for the Processing Service (Modular version)

This module handles:
- Service initialization and lifespan
- CORS configuration
- Inclusion of modular API routes
"""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import modular components
from app.api.threads import router as threads_router
from app.api.files import router as files_router
from app.api.messages import router as messages_router
from app.api.models import router as models_router
from app.db.connection_sqlmodel import init_db, close_db
from app.db.vector import get_vector_db

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Service lifespan management.
    Performs startup tasks like database initialization.
    """
    logger.info("--- RAG Service Starting ---")
    try:
        logger.info("Initializing PostgreSQL database with SQLModel...")
        await init_db()
        logger.info("Database initialization complete.")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}", exc_info=True)
        # We continue to allow the process to start so it can be debugged via /health

    try:
        logger.info("Initializing Weaviate collections...")
        await get_vector_db().ensure_collections()
        logger.info("Weaviate collection initialization complete.")
    except Exception as e:
        logger.critical(f"Failed to initialize Weaviate collections: {e}", exc_info=True)
        
    yield
    logger.info("--- RAG Service Shutting Down ---")
    try:
        logger.info("Closing database connections...")
        await close_db()
        logger.info("Database connections closed.")
    except Exception as e:
        logger.error(f"Error during database shutdown: {e}")

app = FastAPI(
    title="RAG Service",
    description="Modular Retrieval-Augmented Generation Service for AskPDF",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware for cross-service communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register modular routes
app.include_router(threads_router, prefix="/api")
app.include_router(files_router, prefix="/api")
app.include_router(messages_router, prefix="/api")
app.include_router(models_router, prefix="/api")

@app.get("/health")
async def health_check():
    """Service health check endpoint."""
    return {
        "status": "ok",
        "service": "rag-service",
        "version": "2.0.0",
        "mode": "modular"
    }


# Mount static files last to avoid shadowing API routes
app.mount("/files", StaticFiles(directory="/static"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
