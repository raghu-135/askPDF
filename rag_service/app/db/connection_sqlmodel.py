"""
connection_sqlmodel.py - PostgreSQL connection management with SQLModel.

Provides async engine, session factory, and database lifecycle management.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlmodel import SQLModel

logger = logging.getLogger(__name__)

# Database URLs from environment - must be explicitly set
DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL is None:
    raise RuntimeError("DATABASE_URL environment variable is required")

TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL")
if TEST_DATABASE_URL is None:
    raise RuntimeError("TEST_DATABASE_URL environment variable is required")

# Connection pool settings - must be explicitly set
_POSTGRES_POOL_SIZE = os.environ.get("POSTGRES_POOL_SIZE")
if _POSTGRES_POOL_SIZE is None:
    raise RuntimeError("POSTGRES_POOL_SIZE environment variable is required")
POOL_SIZE = int(_POSTGRES_POOL_SIZE)

_POSTGRES_MAX_OVERFLOW = os.environ.get("POSTGRES_MAX_OVERFLOW")
if _POSTGRES_MAX_OVERFLOW is None:
    raise RuntimeError("POSTGRES_MAX_OVERFLOW environment variable is required")
MAX_OVERFLOW = int(_POSTGRES_MAX_OVERFLOW)


def create_engine(database_url: str = None, poolclass=None):
    """Create async engine with proper settings."""
    url = database_url or DATABASE_URL
    
    # Use NullPool for tests to avoid connection issues
    if poolclass == NullPool:
        return create_async_engine(
            url,
            poolclass=NullPool,
            echo=os.environ.get("SQL_ECHO", "false").lower() == "true",
            future=True
        )
    
    return create_async_engine(
        url,
        pool_size=POOL_SIZE,
        max_overflow=MAX_OVERFLOW,
        pool_pre_ping=True,  # Verify connection health before using
        echo=os.environ.get("SQL_ECHO", "false").lower() == "true",
        future=True
    )


# Production engine with connection pooling
engine = create_engine()

# Test engine with NullPool for test isolation
test_engine = create_engine(TEST_DATABASE_URL, poolclass=NullPool)

# Session makers
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

test_session_maker = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


async def init_db(database_url: str = None):
    """Create all tables. Use for development/testing only."""
    url = database_url or DATABASE_URL
    engine = create_engine(url)
    
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
    
    await engine.dispose()
    logger.info("Database tables created")


async def close_db():
    """Close all database connections."""
    await engine.dispose()
    logger.info("Database connections closed")


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a database session with automatic commit/rollback handling."""
    session = async_session_maker()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


@asynccontextmanager
async def get_test_session() -> AsyncGenerator[AsyncSession, None]:
    """Get a test database session."""
    session = test_session_maker()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
