"""
repositories - Database repository modules.

This package contains repository classes for managing database operations
following the repository pattern.
"""

from app.db.repositories.base import BaseRepository
from app.db.repositories.thread_repo import ThreadRepository
from app.db.repositories.file_repo import FileRepository
from app.db.repositories.message_repo import MessageRepository
from app.db.repositories.thread_file_repo import ThreadFileRepository
from app.db.repositories.stats_repo import StatsRepository

# SQLModel repositories (PostgreSQL migration)
# These are imported with try/except for migration compatibility
try:
    from app.db.repositories.thread_repo_sqlmodel import ThreadRepository as ThreadRepositorySQLModel
    from app.db.repositories.file_repo_sqlmodel import FileRepository as FileRepositorySQLModel
    from app.db.repositories.message_repo_sqlmodel import MessageRepository as MessageRepositorySQLModel
    from app.db.repositories.thread_file_repo_sqlmodel import ThreadFileRepository as ThreadFileRepositorySQLModel
    from app.db.repositories.stats_repo_sqlmodel import StatsRepository as StatsRepositorySQLModel

    SQLMODEL_REPOS_AVAILABLE = True
except ImportError:
    SQLMODEL_REPOS_AVAILABLE = False
    ThreadRepositorySQLModel = None
    FileRepositorySQLModel = None
    MessageRepositorySQLModel = None
    ThreadFileRepositorySQLModel = None
    StatsRepositorySQLModel = None

__all__ = [
    "BaseRepository",
    "ThreadRepository",
    "FileRepository",
    "MessageRepository",
    "ThreadFileRepository",
    "StatsRepository",
    # SQLModel repositories (when available)
    "ThreadRepositorySQLModel",
    "FileRepositorySQLModel",
    "MessageRepositorySQLModel",
    "ThreadFileRepositorySQLModel",
    "StatsRepositorySQLModel",
    "SQLMODEL_REPOS_AVAILABLE",
]
