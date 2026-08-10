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
from contextlib import asynccontextmanager, suppress

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
from fastapi.responses import JSONResponse

# Import modular components after logging is configured so app.* loggers emit in Docker.
from app.api.threads import router as threads_router
from app.api.projects import router as projects_router
from app.api.memories import router as memories_router
from app.api.memory_manager import router as memory_manager_router
from app.api.files import router as files_router
from app.api.messages import router as messages_router
from app.api.models import router as models_router
from app.api.agent_workflows import router as agent_workflows_router
from app.api.agent_tasks import router as agent_tasks_router
from app.api.tools import router as tools_router
from app.agent_workflows.repository import AgentWorkflowRepository
from app.db import ensure_default_project
from app.db.connection_sqlmodel import init_db, close_db
from app.db.vector import close_vector_db, get_vector_db
from app.services.memory_service import (
    retry_pending_memory_indexes,
)
from app.services.memory_repair_scheduler import shutdown_memory_repairs
from app.services.embedding_materialization_service import embedding_job_worker
from app.services.agent_task_runtime import run_task_worker


AGENT_TASK_WORKER_SHUTDOWN_GRACE_SECONDS = 30


def _record_agent_task_worker_completion(app: FastAPI, task: asyncio.Task) -> None:
    """Make an unexpected integrated-worker exit immediately observable."""
    if getattr(app.state, "agent_task_worker_status", None) == "stopping":
        app.state.agent_task_worker_status = "stopped"
        return
    app.state.agent_task_worker_status = "failed"
    if task.cancelled():
        logger.critical("Integrated agent task worker was cancelled unexpectedly")
        return
    error = task.exception()
    if error is None:
        logger.critical("Integrated agent task worker exited unexpectedly without an error")
    else:
        logger.critical(
            "Integrated agent task worker exited unexpectedly",
            exc_info=(type(error), error, error.__traceback__),
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
    embedding_job_stop = asyncio.Event()
    embedding_job_task = asyncio.create_task(embedding_job_worker(embedding_job_stop))
    agent_task_worker_stop = asyncio.Event()
    app.state.agent_task_worker_status = "running"
    agent_task_worker = asyncio.create_task(
        run_task_worker(stop_event=agent_task_worker_stop),
        name="agent-task-worker",
    )
    agent_task_worker.add_done_callback(
        lambda task: _record_agent_task_worker_completion(app, task)
    )
    try:
        yield
    finally:
        logger.info("--- RAG Service Shutting Down ---")
        app.state.agent_task_worker_status = "stopping"
        agent_task_worker_stop.set()
        try:
            await asyncio.wait_for(
                asyncio.shield(agent_task_worker),
                timeout=AGENT_TASK_WORKER_SHUTDOWN_GRACE_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Agent task worker exceeded %ss shutdown grace; cancelling active execution",
                AGENT_TASK_WORKER_SHUTDOWN_GRACE_SECONDS,
            )
            agent_task_worker.cancel()
            with suppress(asyncio.CancelledError):
                await agent_task_worker
        except Exception:
            logger.exception("Agent task worker exited unexpectedly")
        memory_maintenance_stop.set()
        await memory_maintenance_task
        embedding_job_stop.set()
        await embedding_job_task
        await shutdown_memory_repairs()
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
app.include_router(memory_manager_router, prefix="/api")
app.include_router(files_router, prefix="/api")
app.include_router(messages_router, prefix="/api")
app.include_router(models_router, prefix="/api")
app.include_router(agent_workflows_router, prefix="/api")
app.include_router(agent_tasks_router, prefix="/api")
app.include_router(tools_router, prefix="/api")

@app.get("/health")
async def health_check():
    """Service health check endpoint."""
    worker_status = getattr(app.state, "agent_task_worker_status", "not_started")
    healthy = worker_status == "running"
    payload = {
        "status": "ok" if healthy else "degraded",
        "service": "rag-service",
        "version": "2.0.0",
        "mode": "modular",
        "agent_task_worker": worker_status,
    }
    return payload if healthy else JSONResponse(status_code=503, content=payload)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
