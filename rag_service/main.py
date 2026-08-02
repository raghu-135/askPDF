"""
main.py - FastAPI entrypoint for the Processing Service (Modular version)

This module handles:
- Service initialization and lifespan
- CORS configuration
- Inclusion of modular API routes
"""

import logging
import os
import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging - LOG_LEVEL must be explicitly set
_log_level_str = os.environ.get("LOG_LEVEL")
if _log_level_str is None:
    raise RuntimeError("LOG_LEVEL environment variable is required")
log_level = _log_level_str.upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logging.getLogger("app").setLevel(getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Import modular components after logging is configured so app.* loggers emit in Docker.
from app.api.threads import router as threads_router
from app.api.projects import router as projects_router
from app.api.memories import router as memories_router
from app.api.files import router as files_router
from app.api.messages import router as messages_router
from app.api.models import router as models_router
from app.api.agent_workflows import router as agent_workflows_router
from app.api.tools import router as tools_router
from app.agent_workflows.repository import AgentWorkflowRepository
from app.db import ensure_default_project
from app.db.connection_sqlmodel import init_db, close_db
from app.db.vector import close_vector_db, get_vector_db
from app.services.memory_service import (
    retry_pending_memory_indexes,
)


async def _memory_maintenance_loop(stop_event: asyncio.Event) -> None:
    """Incrementally retry pending and failed memory indexes."""

    interval = max(30, int(os.environ.get("MEMORY_MAINTENANCE_INTERVAL_SECONDS", "300")))
    batch_size = max(1, min(500, int(os.environ.get("MEMORY_MAINTENANCE_BATCH_SIZE", "100"))))
    while not stop_event.is_set():
        try:
            await retry_pending_memory_indexes(limit=batch_size)
        except Exception:
            logger.exception("Incremental memory maintenance failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass

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
        await ensure_default_project()
        await AgentWorkflowRepository().seed_builtin_workflows()
        logger.info("Database initialization complete.")
    except Exception as e:
        logger.critical(f"Failed to initialize database: {e}", exc_info=True)
        raise

    try:
        logger.info("Initializing Weaviate collections...")
        await get_vector_db().ensure_collections()
        logger.info("Weaviate collection initialization complete.")
    except Exception as e:
        logger.critical(f"Failed to initialize Weaviate collections: {e}", exc_info=True)

    memory_maintenance_stop = asyncio.Event()
    memory_maintenance_task = asyncio.create_task(
        _memory_maintenance_loop(memory_maintenance_stop)
    )
    yield
    logger.info("--- RAG Service Shutting Down ---")
    memory_maintenance_stop.set()
    await memory_maintenance_task
    try:
        logger.info("Closing database connections...")
        await close_db()
        logger.info("Database connections closed.")
    except Exception as e:
        logger.error(f"Error during database shutdown: {e}")
    try:
        logger.info("Closing Weaviate client connection...")
        close_vector_db()
        logger.info("Weaviate client connection closed.")
    except Exception as e:
        logger.error(f"Error during Weaviate shutdown: {e}")

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
app.include_router(projects_router, prefix="/api")
app.include_router(memories_router, prefix="/api")
app.include_router(files_router, prefix="/api")
app.include_router(messages_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(agent_workflows_router, prefix="/api")
app.include_router(tools_router, prefix="/api")

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
