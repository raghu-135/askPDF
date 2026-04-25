"""
connection.py - Database connection management.

This module provides utilities for database initialization and connection management,
including context managers and transaction handling.
"""

import logging
import aiosqlite

from app.db.config import DB_PATH, SCHEMA, MIGRATIONS

logger = logging.getLogger(__name__)


async def init_db() -> None:
    """Initialize the database with the schema and run migrations."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.executescript(SCHEMA)

        # Lightweight migration for existing installations
        for stmt in MIGRATIONS:
            try:
                await db.execute(stmt)
            except aiosqlite.OperationalError as e:
                # Ignore duplicate-column errors for already-migrated DBs
                if "duplicate column name" not in str(e).lower():
                    raise
        await db.commit()
    logger.info(f"Database initialized at {DB_PATH}")


async def get_db():
    """Get a database connection with row factory enabled."""
    db = await aiosqlite.connect(DB_PATH)
    db.row_factory = aiosqlite.Row
    await db.execute("PRAGMA foreign_keys = ON")
    return db
