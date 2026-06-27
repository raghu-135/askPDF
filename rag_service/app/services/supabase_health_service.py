"""Health summary for the parallel Supabase rollout."""

from __future__ import annotations

import os
from typing import Any

from app.db import supabase_client


def _flag(name: str) -> bool:
    return os.getenv(name, "false").lower() == "true"


async def get_supabase_health() -> dict[str, Any]:
    """Return non-secret Supabase rollout status for operators and the UI."""
    flags = {
        "dual_write": _flag("RAG_SUPABASE_DUAL_WRITE"),
        "parity_check": _flag("RAG_SUPABASE_PARITY_CHECK"),
        "threads": _flag("NEXT_PUBLIC_USE_SUPABASE_THREADS"),
        "messages": _flag("NEXT_PUBLIC_USE_SUPABASE_MESSAGES"),
        "files": _flag("NEXT_PUBLIC_USE_SUPABASE_FILES"),
        "realtime": _flag("NEXT_PUBLIC_USE_SUPABASE_REALTIME"),
        "storage": _flag("NEXT_PUBLIC_USE_SUPABASE_STORAGE"),
    }
    reads_enabled = any(flags[key] for key in ("threads", "messages", "files", "realtime", "storage"))
    db_url_configured = bool(supabase_client.get_supabase_db_url())

    db = {
        "configured": db_url_configured,
        "reachable": False,
        "thread_count": None,
        "error": None,
    }
    storage = {
        "enabled": flags["storage"],
        "bucket": os.getenv("SUPABASE_STORAGE_BUCKET", "pdfs"),
        "bucket_exists": None,
        "object_count": None,
        "error": None,
    }

    if db_url_configured and (flags["dual_write"] or flags["parity_check"] or reads_enabled):
        try:
            db["thread_count"] = await supabase_client.fetchval("select count(*) from public.threads")
            db["reachable"] = True
        except Exception as exc:
            db["error"] = str(exc)

        if flags["storage"]:
            try:
                storage["bucket_exists"] = await supabase_client.fetchval(
                    "select exists(select 1 from storage.buckets where id = $1)",
                    storage["bucket"],
                )
                storage["object_count"] = await supabase_client.fetchval(
                    "select count(*) from storage.objects where bucket_id = $1",
                    storage["bucket"],
                )
            except Exception as exc:
                storage["error"] = str(exc)

    enabled = flags["dual_write"] or flags["parity_check"] or reads_enabled
    healthy = (
        enabled
        and db_url_configured
        and bool(db["reachable"])
        and (not flags["storage"] or storage["bucket_exists"] is True)
    )
    status = "healthy" if healthy else "disabled" if not enabled else "degraded"

    return {
        "status": status,
        "healthy": healthy,
        "enabled": enabled,
        "db": db,
        "storage": storage,
        "flags": flags,
    }
