import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.future import select

from app.db.models_sqlmodel import CanonicalDocument, DocumentChunkManifest, DocumentProcessingJob, File

from app.workers import document_conversion_worker as worker


def test_rag_service_lifespan_runs_conversion_jobs_in_process():
    import main as application

    assert application.conversion_job_worker is worker.conversion_job_worker
    assert "conversion_job_worker(conversion_job_stop)" in inspect.getsource(application.lifespan)


@pytest.mark.asyncio
async def test_conversion_job_worker_stops_when_signaled(monkeypatch):
    drain = AsyncMock(return_value=0)
    monkeypatch.setattr(worker, "drain_conversion_jobs", drain)
    stop = asyncio.Event()
    task = asyncio.create_task(worker.conversion_job_worker(stop, interval=0.01))
    await asyncio.sleep(0.03)
    stop.set()
    await asyncio.wait_for(task, timeout=1)
    assert drain.await_count >= 1


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
    job = SimpleNamespace(id="job-1", file_hash="file-1", attempts=1, claim_token="claim-1")
    repo = SimpleNamespace(
        claim_conversion_jobs=AsyncMock(return_value=[job]),
        fail_conversion_job=AsyncMock(),
    )
    monkeypatch.setattr(worker, "get_canonical_document_repo", lambda: repo)
    monkeypatch.setattr(worker, "process_conversion_job", AsyncMock(side_effect=RuntimeError("broken parser")))

    assert await worker.drain_conversion_jobs(limit=1) == 1
    repo.fail_conversion_job.assert_awaited_once()
    assert repo.fail_conversion_job.await_args.args[0] == "job-1"
    assert repo.fail_conversion_job.await_args.args[1] == "claim-1"


@pytest.mark.asyncio
async def test_conversion_repair_requeues_completed_and_exhausted_jobs(engine, monkeypatch):
    from app.db.repositories import canonical_document_repo
    from app.db.repositories.canonical_document_repo import CanonicalDocumentRepository

    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(canonical_document_repo, "async_session_maker", maker)
    file_hash = "b" * 32
    async with maker() as session:
        async with session.begin():
            session.add(File(file_hash=file_hash, file_name="repair.pdf"))

    repo = CanonicalDocumentRepository()
    job = await repo.ensure_conversion_job(
        file_hash=file_hash,
        generation="generation-a",
        extraction_fingerprint="fingerprint-a",
    )
    async with maker() as session:
        async with session.begin():
            row = (await session.execute(select(DocumentProcessingJob).where(DocumentProcessingJob.id == job.id))).scalar_one()
            row.status = "completed"
            row.attempts = 5

    repaired = await repo.ensure_conversion_job(
        file_hash=file_hash,
        generation="generation-a",
        extraction_fingerprint="fingerprint-a",
        force_rebuild=True,
    )
    assert repaired.status == "pending"
    assert repaired.attempts == 0

    async with maker() as session:
        async with session.begin():
            row = (await session.execute(select(DocumentProcessingJob).where(DocumentProcessingJob.id == job.id))).scalar_one()
            row.status = "failed"
            row.attempts = 5

    retried = await repo.ensure_conversion_job(
        file_hash=file_hash,
        generation="generation-a",
        extraction_fingerprint="fingerprint-a",
        retry_failed=True,
    )
    assert retried.status == "pending"
    assert retried.attempts == 0


@pytest.mark.asyncio
async def test_automatic_conversion_reconciliation_preserves_failed_attempts(engine, monkeypatch):
    from app.db.repositories import canonical_document_repo
    from app.db.repositories.canonical_document_repo import CanonicalDocumentRepository

    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(canonical_document_repo, "async_session_maker", maker)
    file_hash = "d" * 32
    async with maker() as session:
        async with session.begin():
            session.add(File(file_hash=file_hash, file_name="failed.pdf"))

    repo = CanonicalDocumentRepository()
    job = await repo.ensure_conversion_job(
        file_hash=file_hash,
        generation="generation-failed",
        extraction_fingerprint="fingerprint-failed",
    )
    async with maker() as session:
        async with session.begin():
            row = (await session.execute(select(DocumentProcessingJob).where(DocumentProcessingJob.id == job.id))).scalar_one()
            row.status = "failed"
            row.attempts = 5
            row.error = "permanent parser failure"

    unchanged = await repo.ensure_conversion_job(
        file_hash=file_hash,
        generation="generation-failed",
        extraction_fingerprint="fingerprint-failed",
    )
    assert unchanged.status == "failed"
    assert unchanged.attempts == 5
    assert unchanged.error == "permanent parser failure"


@pytest.mark.asyncio
async def test_manifest_publication_rejects_stale_generation_and_keeps_current(engine, monkeypatch):
    from app.db.repositories import canonical_document_repo
    from app.db.repositories.canonical_document_repo import CanonicalDocumentRepository

    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    monkeypatch.setattr(canonical_document_repo, "async_session_maker", maker)
    file_hash = "e" * 32
    async with maker() as session:
        async with session.begin():
            session.add(File(file_hash=file_hash, file_name="versioned.pdf"))
            await session.flush()
            session.add(CanonicalDocument(
                file_hash=file_hash,
                generation="generation-new",
                extraction_fingerprint="fingerprint-new",
                status="completed",
                document_json={"schema_version": "test"},
                source_metadata_json={"_file_hash": file_hash},
            ))
            session.add(DocumentChunkManifest(
                manifest_id="manifest-current",
                file_hash=file_hash,
                embedding_model="model-a",
                generation="generation-new",
                extraction_fingerprint="fingerprint-new",
                chunking_fingerprint="chunking-new",
                source_version="version-new",
                status="completed",
                vector_status="completed",
                is_current=True,
                published_at=datetime.now(timezone.utc),
            ))
            session.add(DocumentChunkManifest(
                manifest_id="manifest-stale",
                file_hash=file_hash,
                embedding_model="model-a",
                generation="generation-old",
                extraction_fingerprint="fingerprint-old",
                chunking_fingerprint="chunking-old",
                source_version="version-old",
                status="completed",
                vector_status="completed",
            ))

    result = await CanonicalDocumentRepository().publish_manifest("manifest-stale", vector_count=1)
    assert result.published is False
    assert result.stale is True

    async with maker() as session:
        async with session.begin():
            current = await session.get(DocumentChunkManifest, "manifest-current")
            stale = await session.get(DocumentChunkManifest, "manifest-stale")
            assert current.is_current is True
            assert stale.is_current is False
