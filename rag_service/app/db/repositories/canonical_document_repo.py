"""Persistence for canonical documents and their deterministic projections."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Iterable, Optional
import uuid

from sqlalchemy import delete, select, update

from app.db.connection_sqlmodel import async_session_maker
from app.db.models_sqlmodel import (
    CanonicalDocument,
    DocumentProcessingJob,
    DocumentChunk,
    DocumentChunkManifest,
    DocumentElement,
    DocumentSection,
)
from app.time_utils import utc_now
from app.services.document_pipeline import stable_source_id


class CanonicalDocumentRepository:
    def __init__(self, session=None):
        self._session = session

    async def _owned_session(self):
        return self._session or async_session_maker()

    async def get(self, file_hash: str) -> Optional[CanonicalDocument]:
        session = await self._owned_session()
        async with session.begin():
            result = await session.execute(select(CanonicalDocument).where(CanonicalDocument.file_hash == file_hash))
            return result.scalar_one_or_none()

    async def ensure_conversion_job(
        self,
        *,
        file_hash: str,
        generation: str,
        extraction_fingerprint: str,
        force_rebuild: bool = False,
    ) -> DocumentProcessingJob:
        """Enqueue one durable conversion target without executing conversion in the request."""
        session = await self._owned_session()
        async with session.begin():
            job = (await session.execute(select(DocumentProcessingJob).where(
                DocumentProcessingJob.file_hash == file_hash,
                DocumentProcessingJob.job_kind == "conversion",
                DocumentProcessingJob.embedding_model == "",
                DocumentProcessingJob.generation == generation,
                DocumentProcessingJob.extraction_fingerprint == extraction_fingerprint,
                DocumentProcessingJob.chunking_fingerprint == "",
            ).with_for_update())).scalars().first()
            if job is None:
                job = DocumentProcessingJob(
                    file_hash=file_hash,
                    job_kind="conversion",
                    generation=generation,
                    extraction_fingerprint=extraction_fingerprint,
                    status="pending",
                    available_at=utc_now(),
                )
                session.add(job)
            elif job.status == "failed" or (force_rebuild and job.status == "completed"):
                job.status = "pending"
                job.attempts = 0
                job.available_at = utc_now()
                job.error = None
                job.claimed_at = None
                job.claim_token = None
                job.completed_at = None
                job.updated_at = utc_now()
            await session.flush()
            return job

    async def claim_conversion_jobs(self, *, limit: int = 10, stale_after_seconds: int = 900) -> list[DocumentProcessingJob]:
        now = utc_now()
        stale_cutoff = now - timedelta(seconds=stale_after_seconds)
        session = await self._owned_session()
        async with session.begin():
            await session.execute(update(DocumentProcessingJob).where(
                DocumentProcessingJob.job_kind == "conversion",
                DocumentProcessingJob.status == "running",
                DocumentProcessingJob.claimed_at < stale_cutoff,
            ).values(status="pending", available_at=now, claimed_at=None, claim_token=None, updated_at=now))
            jobs = list((await session.execute(select(DocumentProcessingJob).where(
                DocumentProcessingJob.job_kind == "conversion",
                DocumentProcessingJob.status.in_(("pending", "failed")),
                DocumentProcessingJob.available_at <= now,
                DocumentProcessingJob.attempts < 5,
            ).order_by(DocumentProcessingJob.available_at, DocumentProcessingJob.created_at, DocumentProcessingJob.id).limit(max(1, int(limit))).with_for_update(skip_locked=True))).scalars().all())
            for job in jobs:
                job.status = "running"
                job.attempts = int(job.attempts or 0) + 1
                job.claimed_at = now
                job.claim_token = uuid.uuid4().hex
                job.updated_at = now
                job.error = None
            await session.flush()
            return jobs

    async def fail_conversion_job(self, job_id: str, error: Exception) -> bool:
        now = utc_now()
        session = await self._owned_session()
        async with session.begin():
            row = (await session.execute(select(DocumentProcessingJob).where(
                DocumentProcessingJob.id == job_id,
                DocumentProcessingJob.job_kind == "conversion",
                DocumentProcessingJob.status == "running",
            ).with_for_update())).scalar_one_or_none()
            if row is None:
                return False
            delay = min(300, 2 ** max(0, int(row.attempts or 1)))
            row.status = "failed"
            row.error = str(error)[:2000]
            row.claimed_at = None
            row.available_at = now + timedelta(seconds=delay)
            row.updated_at = now
            await session.flush()
            return True

    async def claim_conversion(self, file_hash: str, fingerprint: str, generation: str, *, stale_after_seconds: int = 900, force_rebuild: bool = False, claim_token: str | None = None) -> str | None:
        """Return a claim token, or None when another/current result owns work."""
        session = await self._owned_session()
        async with session.begin():
            result = await session.execute(select(CanonicalDocument).where(CanonicalDocument.file_hash == file_hash).with_for_update())
            row = result.scalar_one_or_none()
            now = utc_now()
            job = (await session.execute(select(DocumentProcessingJob).where(
                DocumentProcessingJob.file_hash == file_hash,
                DocumentProcessingJob.job_kind == "conversion",
                DocumentProcessingJob.embedding_model == "",
                DocumentProcessingJob.generation == generation,
                DocumentProcessingJob.extraction_fingerprint == fingerprint,
                DocumentProcessingJob.chunking_fingerprint == "",
            ).with_for_update())).scalars().first()
            if row is not None and row.extraction_fingerprint == fingerprint and row.status == "completed" and not force_rebuild:
                if job is not None and job.status != "completed":
                    job.status = "completed"
                    job.claim_token = None
                    job.completed_at = row.completed_at or now
                return None
            worker_claim = bool(claim_token and job is not None and job.status == "running" and job.claim_token == claim_token)
            if row is not None and row.status == "running" and not worker_claim:
                age = (now - (row.claimed_at or row.created_at)).total_seconds() if (row.claimed_at or row.created_at) else 0
                if age < stale_after_seconds:
                    return None
            if job is None:
                job = DocumentProcessingJob(
                    file_hash=file_hash, job_kind="conversion", generation=generation,
                    extraction_fingerprint=fingerprint,
                )
                session.add(job)
            if not worker_claim:
                if force_rebuild and job.status in {"completed", "failed"}:
                    job.attempts = 0
                job.status = "running"
                claim_token = claim_token or uuid.uuid4().hex
                job.claim_token = claim_token
                job.attempts = int(job.attempts or 0) + 1
            job.claimed_at = now
            job.error = None
            job.updated_at = now
            if row is None:
                row = CanonicalDocument(file_hash=file_hash, generation=generation, extraction_fingerprint=fingerprint, status="running")
                session.add(row)
            else:
                row.generation = generation
                row.extraction_fingerprint = fingerprint
                row.status = "running"
                row.document_json = {}
                row.source_metadata_json = {}
                row.failure_json = None
                row.completed_at = None
                row.created_at = now
            row.claim_token = claim_token
            row.claimed_at = now
            await session.flush()
            return claim_token

    async def renew_conversion_claim(self, file_hash: str, claim_token: str) -> bool:
        session = await self._owned_session()
        async with session.begin():
            now = utc_now()
            result = await session.execute(
                update(CanonicalDocument)
                .where(CanonicalDocument.file_hash == file_hash, CanonicalDocument.status == "running", CanonicalDocument.claim_token == claim_token)
                .values(claim_token=claim_token, claimed_at=now)
            )
            if not result.rowcount:
                return False
            await session.execute(
                update(DocumentProcessingJob)
                .where(DocumentProcessingJob.file_hash == file_hash, DocumentProcessingJob.job_kind == "conversion", DocumentProcessingJob.status == "running", DocumentProcessingJob.claim_token == claim_token)
                .values(claimed_at=now, updated_at=now)
            )
            return True

    async def complete_conversion(self, *, file_hash: str, claim_token: str, generation: str, fingerprint: str, docling_version: str, document_json: dict[str, Any], source_metadata: dict[str, Any], sections: Iterable[dict[str, Any]], elements: Iterable[dict[str, Any]]) -> bool:
        session = await self._owned_session()
        async with session.begin():
            result = await session.execute(select(CanonicalDocument).where(CanonicalDocument.file_hash == file_hash).with_for_update())
            row = result.scalar_one_or_none()
            if row is None or row.status != "running" or row.claim_token != claim_token:
                return False
            row.generation = generation
            row.extraction_fingerprint = fingerprint
            row.docling_version = docling_version
            row.document_json = document_json
            row.source_metadata_json = source_metadata
            row.failure_json = None
            row.status = "completed"
            row.claim_token = None
            row.claimed_at = None
            row.completed_at = utc_now()
            job = (await session.execute(select(DocumentProcessingJob).where(
                DocumentProcessingJob.file_hash == file_hash,
                DocumentProcessingJob.job_kind == "conversion",
                DocumentProcessingJob.generation == generation,
                DocumentProcessingJob.extraction_fingerprint == fingerprint,
                DocumentProcessingJob.chunking_fingerprint == "",
                DocumentProcessingJob.claim_token == claim_token,
            ).with_for_update())).scalars().first()
            if job is not None:
                job.status = "completed"
                job.claimed_at = None
                job.claim_token = None
                job.completed_at = row.completed_at
                job.updated_at = row.completed_at
            await session.execute(delete(DocumentSection).where(DocumentSection.file_hash == file_hash))
            await session.execute(delete(DocumentElement).where(DocumentElement.file_hash == file_hash))
            for section in sections:
                pages = list(section.get("pages") or [])
                session.add(DocumentSection(
                    section_id=str(section["section_id"]), file_hash=file_hash, generation=generation,
                    parent_section_id=section.get("parent_section_id"), section_order=int(section.get("section_order", 0)),
                    level=int(section.get("level", 0)), title=str(section.get("title") or ""),
                    heading_path=list(section.get("heading_path") or []), element_ids=list(section.get("element_ids") or []),
                    page_start=min(pages) if pages else None, page_end=max(pages) if pages else None,
                ))
            for element in elements:
                pages = list(element.get("pages") or [])
                session.add(DocumentElement(
                    element_id=str(element["element_id"]), file_hash=file_hash, generation=generation,
                    element_order=int(element.get("element_order", 0)), element_type=str(element.get("element_type") or "unspecified"),
                    label=element.get("label"), text=str(element.get("text") or ""), section_id=element.get("section_id"),
                    parent_element_id=element.get("parent_element_id") or element.get("parent_ref"), page_start=min(pages) if pages else None, page_end=max(pages) if pages else None,
                    provenance_json={"items": list(element.get("provenance") or [])}, structure_json={"raw": element.get("raw") or {}, "heading_path": list(element.get("heading_path") or []), "table_structure": element.get("table_structure")},
                ))
            await session.flush()
            return True

    async def fail_conversion(self, file_hash: str, error: dict[str, Any], claim_token: str | None = None) -> bool:
        session = await self._owned_session()
        async with session.begin():
            now = utc_now()
            query = update(CanonicalDocument).where(CanonicalDocument.file_hash == file_hash, CanonicalDocument.status == "running")
            if claim_token is not None:
                query = query.where(CanonicalDocument.claim_token == claim_token)
            result = await session.execute(query.values(status="failed", failure_json=error, completed_at=None, claim_token=None, claimed_at=None))
            job_query = select(DocumentProcessingJob).where(
                DocumentProcessingJob.file_hash == file_hash,
                DocumentProcessingJob.job_kind == "conversion",
                DocumentProcessingJob.status == "running",
            ).with_for_update()
            if claim_token is not None:
                job_query = job_query.where(DocumentProcessingJob.claim_token == claim_token)
            jobs = (await session.execute(job_query)).scalars().all()
            for job in jobs:
                delay = min(300, 2 ** max(0, int(job.attempts or 1)))
                job.status = "failed"
                job.error = str(error)[:2000]
                job.claimed_at = None
                job.claim_token = None
                job.available_at = now + timedelta(seconds=delay)
                job.updated_at = now
            return bool(result.rowcount)

    async def get_sections(self, file_hash: str, generation: str | None = None) -> list[DocumentSection]:
        session = await self._owned_session()
        async with session.begin():
            query = select(DocumentSection).where(DocumentSection.file_hash == file_hash).order_by(DocumentSection.section_order)
            if generation:
                query = query.where(DocumentSection.generation == generation)
            return list((await session.execute(query)).scalars().all())

    async def get_descendant_section_ids(self, file_hash: str, generation: str, section_id: str) -> list[str]:
        """Return a section and all nested sections in document order."""
        sections = await self.get_sections(file_hash, generation)
        children: dict[str | None, list[str]] = {}
        for section in sections:
            children.setdefault(section.parent_section_id, []).append(str(section.section_id))
        descendants: list[str] = []
        pending = [str(section_id)]
        while pending:
            current = pending.pop(0)
            if current in descendants:
                continue
            descendants.append(current)
            pending[0:0] = children.get(current, [])
        order = {str(section.section_id): index for index, section in enumerate(sections)}
        return sorted(descendants, key=lambda value: order.get(value, len(order)))

    async def get_elements(self, file_hash: str, generation: str | None = None) -> list[DocumentElement]:
        session = await self._owned_session()
        async with session.begin():
            query = select(DocumentElement).where(DocumentElement.file_hash == file_hash).order_by(DocumentElement.element_order)
            if generation:
                query = query.where(DocumentElement.generation == generation)
            return list((await session.execute(query)).scalars().all())

    async def replace_manifest(self, *, file_hash: str, embedding_model: str, generation: str, chunking_fingerprint: str, chunks: Iterable[dict[str, Any]]) -> DocumentChunkManifest:
        chunk_list = list(chunks)
        session = await self._owned_session()
        async with session.begin():
            # Keep old manifests and their chunks intact while a replacement is
            # staged. Search remains pinned to the previously published set if
            # this process dies before publication.
            manifest = DocumentChunkManifest(
                file_hash=file_hash,
                embedding_model=embedding_model,
                generation=generation,
                chunking_fingerprint=chunking_fingerprint,
                status="running",
                vector_status="missing",
                vector_count=0,
                expected_chunk_count=len(chunk_list),
                expected_chunk_ids=[str(item["chunk_id"]) for item in chunk_list],
                expected_source_ids=[stable_source_id(file_hash, generation, str(item["chunk_id"])) for item in chunk_list],
            )
            session.add(manifest)
            projection_job = (await session.execute(select(DocumentProcessingJob).where(
                DocumentProcessingJob.file_hash == file_hash,
                DocumentProcessingJob.job_kind == "projection",
                DocumentProcessingJob.embedding_model == embedding_model,
                DocumentProcessingJob.generation == generation,
                DocumentProcessingJob.extraction_fingerprint == chunking_fingerprint,
                DocumentProcessingJob.chunking_fingerprint == chunking_fingerprint,
            ).with_for_update())).scalars().first()
            if projection_job is None:
                projection_job = DocumentProcessingJob(
                    file_hash=file_hash, job_kind="projection", embedding_model=embedding_model,
                    generation=generation, extraction_fingerprint=chunking_fingerprint,
                    chunking_fingerprint=chunking_fingerprint, status="running", attempts=1,
                    claimed_at=utc_now(),
                )
                session.add(projection_job)
            else:
                projection_job.status = "running"
                projection_job.attempts = int(projection_job.attempts or 0) + 1
                projection_job.claimed_at = utc_now()
                projection_job.error = None
                projection_job.updated_at = utc_now()
            await session.flush()
            for item in chunk_list:
                pages = list(item.get("pages") or [])
                session.add(DocumentChunk(
                    chunk_id=str(item["chunk_id"]), manifest_id=manifest.manifest_id, file_hash=file_hash, embedding_model=embedding_model,
                    source_id=stable_source_id(file_hash, generation, str(item["chunk_id"])),
                    chunk_order=int(item.get("chunk_order", 0)), body_text=str(item.get("body_text") or ""), contextualized_text=str(item.get("contextualized_text") or ""),
                    sentence_ids=list(item.get("sentence_ids") or []), source_element_ids=list(item.get("source_element_ids") or []), section_id=item.get("section_id"), table_id=item.get("table_id"),
                    page_start=min(pages) if pages else None, page_end=max(pages) if pages else None,
                    metadata_json={
                        "pages": pages,
                        "heading_path": list(item.get("heading_path") or []),
                        "token_count": item.get("token_count"),
                        "source_spans": list(item.get("source_spans") or []),
                        "tags": list(item.get("tags") or []),
                        "tag_provenance": dict(item.get("tag_provenance") or {}),
                    },
                ))
            manifest.status = "completed"
            manifest.completed_at = utc_now()
            projection_job.status = "completed"
            projection_job.claimed_at = None
            projection_job.completed_at = manifest.completed_at
            projection_job.updated_at = manifest.completed_at
            await session.flush()
            return manifest

    async def get_manifest(self, file_hash: str, embedding_model: str, generation: str, chunking_fingerprint: str) -> Optional[DocumentChunkManifest]:
        session = await self._owned_session()
        async with session.begin():
            query = select(DocumentChunkManifest).where(
                DocumentChunkManifest.file_hash == file_hash,
                DocumentChunkManifest.embedding_model == embedding_model,
                DocumentChunkManifest.generation == generation,
                DocumentChunkManifest.chunking_fingerprint == chunking_fingerprint,
            )
            return (await session.execute(query.order_by(DocumentChunkManifest.created_at.desc()))).scalars().first()

    async def mark_vector_status(self, manifest_id: str, status: str, *, vector_count: int | None = None, failure: dict[str, Any] | None = None) -> bool:
        session = await self._owned_session()
        async with session.begin():
            result = await session.execute(select(DocumentChunkManifest).where(DocumentChunkManifest.manifest_id == manifest_id).with_for_update())
            manifest = result.scalar_one_or_none()
            if manifest is None:
                return False
            manifest.vector_status = status
            if vector_count is not None:
                manifest.vector_count = int(vector_count)
            manifest.failure_json = failure
            await session.flush()
            return True

    async def publish_manifest(self, manifest_id: str, *, vector_count: int) -> bool:
        session = await self._owned_session()
        async with session.begin():
            manifest = (await session.execute(
                select(DocumentChunkManifest).where(DocumentChunkManifest.manifest_id == manifest_id).with_for_update()
            )).scalar_one_or_none()
            if manifest is None:
                return False
            now = utc_now()
            await session.execute(
                update(DocumentChunkManifest)
                .where(
                    DocumentChunkManifest.file_hash == manifest.file_hash,
                    DocumentChunkManifest.embedding_model == manifest.embedding_model,
                    DocumentChunkManifest.published_at.is_not(None),
                    DocumentChunkManifest.manifest_id != manifest_id,
                )
                .values(superseded_at=now)
            )
            manifest.vector_status = "completed"
            manifest.vector_count = int(vector_count)
            manifest.published_at = now
            manifest.superseded_at = None
            manifest.failure_json = None
            await session.flush()
            return True

    async def get_published_manifest_ids(self, file_hash: str, embedding_model: str, *, exclude_manifest_id: str | None = None) -> list[str]:
        session = await self._owned_session()
        async with session.begin():
            query = select(DocumentChunkManifest.manifest_id).where(
                DocumentChunkManifest.file_hash == file_hash,
                DocumentChunkManifest.embedding_model == embedding_model,
                DocumentChunkManifest.published_at.is_not(None),
            )
            if exclude_manifest_id:
                query = query.where(DocumentChunkManifest.manifest_id != exclude_manifest_id)
            return [str(value) for value in (await session.execute(query)).scalars().all()]

    async def delete_manifest(self, manifest_id: str) -> bool:
        session = await self._owned_session()
        async with session.begin():
            result = await session.execute(delete(DocumentChunkManifest).where(DocumentChunkManifest.manifest_id == manifest_id))
            return bool(result.rowcount)

    async def get_ready_manifest(self, file_hash: str, embedding_model: str, generation: str, chunking_fingerprint: str) -> Optional[DocumentChunkManifest]:
        session = await self._owned_session()
        async with session.begin():
            query = select(DocumentChunkManifest).where(
                DocumentChunkManifest.file_hash == file_hash,
                DocumentChunkManifest.embedding_model == embedding_model,
                DocumentChunkManifest.generation == generation,
                DocumentChunkManifest.chunking_fingerprint == chunking_fingerprint,
                DocumentChunkManifest.status == "completed",
                DocumentChunkManifest.vector_status == "completed",
                DocumentChunkManifest.published_at.is_not(None),
                DocumentChunkManifest.superseded_at.is_(None),
            )
            return (await session.execute(query)).scalars().first()

    async def get_chunks(self, file_hash: str, embedding_model: str, *, generation: str | None = None, manifest_id: str | None = None, section_id: str | None = None, section_ids: set[str] | None = None, table_id: str | None = None, chunk_ids: set[str] | None = None) -> list[DocumentChunk]:
        session = await self._owned_session()
        async with session.begin():
            query = select(DocumentChunk).join(DocumentChunkManifest, DocumentChunk.manifest_id == DocumentChunkManifest.manifest_id).where(DocumentChunk.file_hash == file_hash, DocumentChunk.embedding_model == embedding_model)
            if manifest_id:
                query = query.where(DocumentChunk.manifest_id == manifest_id)
            else:
                query = query.where(DocumentChunkManifest.status == "completed", DocumentChunkManifest.published_at.is_not(None), DocumentChunkManifest.superseded_at.is_(None))
            if generation:
                query = query.where(DocumentChunkManifest.generation == generation)
            if section_ids:
                query = query.where(DocumentChunk.section_id.in_(section_ids))
            elif section_id:
                query = query.where(DocumentChunk.section_id == section_id)
            if table_id:
                query = query.where(DocumentChunk.table_id == table_id)
            if chunk_ids:
                query = query.where(DocumentChunk.chunk_id.in_(chunk_ids))
            return list((await session.execute(query.order_by(DocumentChunk.chunk_order, DocumentChunk.chunk_id))).scalars().all())

    async def get_chunks_by_source_id(self, source_id: str, embedding_model: str, file_hash: str | None = None, manifest_id: str | None = None) -> list[DocumentChunk]:
        session = await self._owned_session()
        async with session.begin():
            query = select(DocumentChunk).join(DocumentChunkManifest, DocumentChunk.manifest_id == DocumentChunkManifest.manifest_id).where(
                DocumentChunk.source_id == source_id,
                DocumentChunk.embedding_model == embedding_model,
            )
            if manifest_id:
                query = query.where(DocumentChunk.manifest_id == manifest_id)
            else:
                query = query.where(
                    DocumentChunkManifest.status == "completed",
                    DocumentChunkManifest.vector_status == "completed",
                    DocumentChunkManifest.published_at.is_not(None),
                    DocumentChunkManifest.superseded_at.is_(None),
                )
            if file_hash:
                query = query.where(DocumentChunk.file_hash == file_hash)
            return list((await session.execute(query.order_by(DocumentChunkManifest.published_at.desc(), DocumentChunk.chunk_order))).scalars().all())


_repository: CanonicalDocumentRepository | None = None


def get_canonical_document_repo() -> CanonicalDocumentRepository:
    global _repository
    if _repository is None:
        _repository = CanonicalDocumentRepository()
    return _repository


__all__ = ["CanonicalDocumentRepository", "get_canonical_document_repo"]
