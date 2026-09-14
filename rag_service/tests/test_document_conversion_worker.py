from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.workers import document_conversion_worker as worker


@pytest.mark.asyncio
async def test_conversion_worker_processes_and_publishes_durable_job(monkeypatch):
    file_hash = "a" * 32
    job = SimpleNamespace(id="job-1", file_hash=file_hash, attempts=1)
    file = SimpleNamespace(file_name="report.pdf")
    store = SimpleNamespace(exists=AsyncMock(return_value=True), read=AsyncMock(return_value=b"pdf"))
    parsed = {"version": "2.0", "sentences": []}

    monkeypatch.setattr(worker, "get_file", AsyncMock(return_value=file))
    monkeypatch.setattr(worker, "get_content_store", lambda: store)
    monkeypatch.setattr(worker, "convert_pdf_and_project", AsyncMock(return_value=parsed))
    publish = AsyncMock(return_value=True)
    monkeypatch.setattr(worker, "publish_reading_projection", publish)
    status = AsyncMock()
    monkeypatch.setattr(worker, "update_parsing_status", status)

    await worker.process_conversion_job(job)

    worker.convert_pdf_and_project.assert_awaited_once_with(
        file_hash=file_hash,
        data=b"pdf",
        file_name="report.pdf",
        source_metadata={"original_title": "report.pdf"},
    )
    publish.assert_awaited_once_with(file_hash, parsed)
    status.assert_awaited_once_with(file_hash, "completed")


@pytest.mark.asyncio
async def test_conversion_worker_records_retryable_failure(monkeypatch):
    job = SimpleNamespace(id="job-1", file_hash="file-1", attempts=1)
    repo = SimpleNamespace(
        claim_conversion_jobs=AsyncMock(return_value=[job]),
        fail_conversion_job=AsyncMock(),
    )
    monkeypatch.setattr(worker, "get_canonical_document_repo", lambda: repo)
    monkeypatch.setattr(worker, "process_conversion_job", AsyncMock(side_effect=RuntimeError("broken parser")))

    assert await worker.drain_conversion_jobs(limit=1) == 1
    repo.fail_conversion_job.assert_awaited_once()
    assert repo.fail_conversion_job.await_args.args[0] == "job-1"
