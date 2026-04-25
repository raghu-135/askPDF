"""
base.py - Base repository with common database operation patterns.

This module provides a base repository class that encapsulates common database
operations and patterns to reduce code duplication across repositories.
"""

import logging
from typing import Optional, List, Dict, Any, Tuple
import aiosqlite

from app.db.config import DB_PATH

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common database operation helpers."""

    def __init__(self):
        """Initialize the base repository."""
        self.db_path = DB_PATH

    async def _execute(
        self,
        sql: str,
        params: Optional[Tuple] = None,
        *,
        commit: bool = True
    ) -> aiosqlite.Cursor:
        """Execute a SQL statement with optional parameters."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            cursor = await db.execute(sql, params or ())
            if commit:
                await db.commit()
            return cursor

    async def _fetch_one(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> Optional[aiosqlite.Row]:
        """Fetch a single row from the database."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            cursor = await db.execute(sql, params or ())
            return await cursor.fetchone()

    async def _fetch_all(
        self,
        sql: str,
        params: Optional[Tuple] = None
    ) -> List[aiosqlite.Row]:
        """Fetch all rows from the database."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            cursor = await db.execute(sql, params or ())
            return await cursor.fetchall()

    async def _transaction(self, callback) -> Any:
        """Execute a callback within a database transaction."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA foreign_keys = ON")
            await db.execute("BEGIN IMMEDIATE")
            try:
                result = await callback(db)
                await db.commit()
                return result
            except Exception:
                await db.rollback()
                raise
