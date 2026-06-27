"""Best-effort Supabase Storage mirroring for local PDF files."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

from app.db import supabase_client

logger = logging.getLogger(__name__)


def storage_enabled() -> bool:
    return os.getenv("NEXT_PUBLIC_USE_SUPABASE_STORAGE", "false").lower() == "true"


def storage_url() -> str:
    return os.getenv("SUPABASE_STORAGE_URL", "http://supabase-kong:8000/storage/v1").rstrip("/")


def storage_bucket() -> str:
    return os.getenv("SUPABASE_STORAGE_BUCKET", "pdfs")


async def mirror_pdf_to_storage(file_hash: str, source_path: str | Path) -> None:
    """Upload a local PDF into Supabase Storage and record object metadata.

    This is intentionally best-effort. FastAPI static storage remains the primary
    operational fallback, so Storage failures should be observable but non-blocking.
    """
    if not storage_enabled():
        return

    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not service_key:
        logger.warning("Supabase Storage mirror skipped for %s: missing service-role key", file_hash)
        return

    path = Path(source_path)
    if not path.exists():
        logger.warning("Supabase Storage mirror skipped for %s: local PDF missing at %s", file_hash, path)
        return

    bucket = storage_bucket()
    object_path = f"{file_hash}.pdf"
    headers = {
        "authorization": f"Bearer {service_key}",
        "apikey": service_key,
        "content-type": "application/pdf",
        "x-upsert": "true",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{storage_url()}/object/{bucket}/{object_path}",
                headers=headers,
                content=path.read_bytes(),
            )
            response.raise_for_status()

        await supabase_client.execute(
            """
            update public.files
            set storage_bucket = $2, storage_path = $3
            where file_hash = $1
            """,
            file_hash,
            bucket,
            object_path,
        )
    except Exception as exc:
        logger.warning("Supabase Storage mirror skipped for %s: %s", file_hash, exc)
