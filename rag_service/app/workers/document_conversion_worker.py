"""In-process drain loop for durable PDF conversion jobs."""

from __future__ import annotations

import asyncio
import logging

from app.db import get_file, update_parsing_status
from app.db.repositories.canonical_document_repo import get_canonical_document_repo
from app.services.content_store import get_content_store, pdf_content_key


logger = logging.getLogger(__name__)


async def convert_pdf_and_project(*args, **kwargs):
    """Load the conversion stack only when a worker has a job to process."""
    from app.services.document_conversion_service import convert_pdf_and_project as convert

    return await convert(*args, **kwargs)


async def publish_reading_projection(*args, **kwargs):
    """Load the legacy reading-cache publisher only when needed."""
    from app.services.file_processing_service import publish_reading_projection as publish

    return await publish(*args, **kwargs)


async def process_conversion_job(job) -> None:
    file = await get_file(job.file_hash)
    if file is None:
        raise FileNotFoundError(f"File record not found for {job.file_hash}")
    store = get_content_store()
    key = pdf_content_key(job.file_hash)
    if not await store.exists(key):
        raise FileNotFoundError(f"PDF content not found for {job.file_hash}")
    data = await store.read(key)
    conversion_kwargs = {
        "file_hash": job.file_hash,
        "data": data,
        "file_name": file.file_name,
        "source_metadata": {"original_title": file.file_name},
    }
    if getattr(job, "claim_token", None):
        conversion_kwargs["claim_token"] = job.claim_token
    parsed = await convert_pdf_and_project(**conversion_kwargs)
    if not await publish_reading_projection(job.file_hash, parsed):
        raise RuntimeError(f"Could not publish reading projection for {job.file_hash}")
    await update_parsing_status(job.file_hash, "completed")


async def drain_conversion_jobs(*, limit: int = 10) -> int:
    repo = get_canonical_document_repo()
    jobs = await repo.claim_conversion_jobs(limit=limit)
    for job in jobs:
        try:
            await process_conversion_job(job)
        except Exception as exc:
            logger.warning(
                "Conversion job failed | id=%s file=%s attempt=%s: %s",
                job.id,
                job.file_hash,
                job.attempts,
                exc,
                exc_info=True,
            )
            await repo.fail_conversion_job(job.id, job.claim_token, exc)
    return len(jobs)


async def conversion_job_worker(stop_event: asyncio.Event, *, interval: float = 2.0) -> None:
    while not stop_event.is_set():
        processed = await drain_conversion_jobs()
        if processed:
            continue
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


