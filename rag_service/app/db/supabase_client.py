"""Small asyncpg client for service-role Supabase database writes."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import asyncpg

logger = logging.getLogger(__name__)

_pool: asyncpg.Pool | None = None


def supabase_enabled() -> bool:
    return os.getenv("RAG_SUPABASE_DUAL_WRITE", "false").lower() == "true"


def parity_enabled() -> bool:
    return os.getenv("RAG_SUPABASE_PARITY_CHECK", "false").lower() == "true"


def get_supabase_db_url() -> str | None:
    return (
        os.getenv("RAG_SUPABASE_DB_URL")
        or os.getenv("SUPABASE_DB_URL")
        or os.getenv("TARGET_DATABASE_URL")
        or os.getenv("MIGRATION_TARGET_DATABASE_URL")
    )


def normalize_db_url(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://", 1)


def json_arg(value: Any) -> str:
    return json.dumps(value if value is not None else {}, default=str)


async def get_pool() -> asyncpg.Pool | None:
    global _pool
    if not supabase_enabled() and not parity_enabled():
        return None
    if _pool is not None:
        return _pool
    url = get_supabase_db_url()
    if not url:
        logger.warning("Supabase dual-write/parity enabled but no Supabase DB URL is configured")
        return None
    _pool = await asyncpg.create_pool(normalize_db_url(url), min_size=1, max_size=3)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


async def execute(sql: str, *args: Any) -> str | None:
    pool = await get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        return await conn.execute(sql, *args)


async def fetchrow(sql: str, *args: Any) -> asyncpg.Record | None:
    pool = await get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        return await conn.fetchrow(sql, *args)


async def fetchval(sql: str, *args: Any) -> Any:
    pool = await get_pool()
    if pool is None:
        return None
    async with pool.acquire() as conn:
        return await conn.fetchval(sql, *args)
