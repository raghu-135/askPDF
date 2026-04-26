"""
API module for RAG Service.

This module provides modular routing organized by domain under /api/* prefix:
- threads.py: Thread CRUD, settings, prompt tools
- files.py: File upload, download, status, annotations
- messages.py: Messages and chat
- models.py: Model availability and health checks

Note: Legacy routes_backup.py exists for reference but is no longer used.
"""

# Modular routers (use /api prefix in main.py)
from app.api.threads import router as threads_router
from app.api.files import router as files_router
from app.api.messages import router as messages_router
from app.api.models import router as models_router

__all__ = [
    "threads_router",
    "files_router",
    "messages_router",
    "models_router",
]
