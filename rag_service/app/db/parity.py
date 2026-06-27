"""Optional parity checks between primary SQLModel Postgres and Supabase."""

from __future__ import annotations

import json
import logging
from typing import Any

from app.db import supabase_client

logger = logging.getLogger(__name__)


async def check_count(label: str, source_count: int, sql: str, *args: Any) -> None:
    if not supabase_client.parity_enabled():
        return
    try:
        target_count = await supabase_client.fetchval(sql, *args)
        if target_count is not None and int(target_count) != int(source_count):
            logger.warning("Supabase parity mismatch for %s: source=%s target=%s", label, source_count, target_count)
    except Exception as exc:
        logger.warning("Supabase parity check failed for %s: %s", label, exc)


async def check_json(label: str, source_payload: Any, sql: str, *args: Any) -> None:
    if not supabase_client.parity_enabled():
        return
    try:
        target_payload = await supabase_client.fetchval(sql, *args)
        normalize = lambda value: json.dumps(value, sort_keys=True, default=str)
        if target_payload is not None and normalize(target_payload) != normalize(source_payload):
            logger.warning("Supabase parity JSON mismatch for %s", label)
    except Exception as exc:
        logger.warning("Supabase parity JSON check failed for %s: %s", label, exc)
