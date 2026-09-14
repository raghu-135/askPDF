"""Hierarchical document retrieval tools over canonical projections."""

from __future__ import annotations

import base64
import json
import re
from collections import defaultdict
from typing import Any

from app.agent.tool_contract import ToolWarningCode, make_tool_error_result, make_tool_result, tool_started
from app.db.repositories.canonical_document_repo import get_canonical_document_repo
from app.rag.retrieval import bounded_retrieval_text
from app.tools.contracts import InspectDocumentRequest, ReadContextRequest, SearchKnowledgeRequest
from app.tools.context import ToolInvocationContext
from app.tools.services import DefaultToolServices, get_tool_services
from app.services.embedding_tokenizer import EmbeddingTokenizerUnavailableError, resolve_embedding_tokenizer


def _cursor_offset(value: str | None) -> int:
    if not value:
        return 0
    try:
        decoded = json.loads(base64.urlsafe_b64decode(value.encode("ascii")).decode("utf-8"))
        return max(0, int(decoded.get("offset", 0)))
    except Exception:
        raise ValueError("invalid continuation cursor")


def _cursor(offset: int) -> str:
    return base64.urlsafe_b64encode(json.dumps({"offset": max(0, offset)}, separators=(",", ":")).encode()).decode()


def _context_cursor(manifest_id: str, anchor_chunk_id: str, expansion: str, segment_index: int) -> str:
    payload = {
        "version": 1,
        "kind": "read_context",
        "manifest_id": manifest_id,
        "anchor_chunk_id": anchor_chunk_id,
        "expansion": expansion,
        "segment_index": max(0, segment_index),
    }
    return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()


def _context_cursor_payload(value: str | None) -> dict[str, Any]:
    if not value:
        return {"segment_index": 0}
    try:
        decoded = json.loads(base64.urlsafe_b64decode(value.encode("ascii")).decode("utf-8"))
    except Exception as exc:
        raise ValueError("invalid context continuation cursor") from exc
    if not isinstance(decoded, dict) or decoded.get("kind") != "read_context" or decoded.get("version") != 1:
        raise ValueError("invalid context continuation cursor")
    decoded["segment_index"] = max(0, int(decoded.get("segment_index", 0)))
    return decoded


_PAGE_REFERENCE_RE = re.compile(r"\bpages?\s+(\d+)(?:\s*(?:-|–|to)\s*(\d+))?\b", re.IGNORECASE)


def _explicit_page_filter(query: str) -> list[int]:
    """Resolve explicit page references before semantic ranking can hide them."""
    pages: set[int] = set()
    for start, end in _PAGE_REFERENCE_RE.findall(str(query or "")):
        first = int(start)
        last = int(end or start)
        if last < first:
            first, last = last, first
        pages.update(range(first, min(last, first + 99) + 1))
    return sorted(pages)


def _normalise_pages(value: Any) -> list[int]:
    values = [value] if isinstance(value, str) else list(value or []) if isinstance(value, (list, tuple, set)) else []
    pages: set[int] = set()
    for item in values:
        if isinstance(item, int) or (isinstance(item, str) and item.strip().isdigit()):
            page = int(item)
            if page > 0:
                pages.add(page)
            continue
        for start, end in re.findall(r"(\d+)\s*(?:-|–|to)\s*(\d+)", str(item), re.IGNORECASE):
            first, last = sorted((int(start), int(end)))
            pages.update(range(first, min(last, first + 999) + 1))
        if not re.search(r"\d+\s*(?:-|–|to)\s*\d+", str(item)):
            pages.update(int(part) for part in re.findall(r"\d+", str(item)) if int(part) > 0)
    return sorted(pages)


def _source_from_chunk(chunk: Any, *, score: Any = None, role: str = "evidence") -> dict[str, Any]:
    metadata = dict(getattr(chunk, "metadata_json", None) or {}) if not isinstance(chunk, dict) else dict(chunk.get("metadata") or {})
    get = chunk.get if isinstance(chunk, dict) else lambda key, default=None: getattr(chunk, key, default)
    pages = _normalise_pages(metadata.get("pages") or get("pages", []))
    chunk_id = str(get("chunk_id", "") or "")
    source_id = str(get("source_id", "") or get("chunk_identity", "") or "")
    if not source_id and not isinstance(chunk, dict):
        source_id = _stable_chunk_source_id(chunk, str(get("embedding_model", "") or ""))
    source_id = source_id or chunk_id
    body = str(metadata.get("body_text") or get("body_text", get("text", "")) or "")
    return {
        "source_id": source_id,
        "file_hash": get("file_hash"),
        "parent_id": get("section_id"),
        "chunk_id": get("chunk_id"),
        "text": body[:200] + ("..." if len(body) > 200 else ""),
        "page_start": get("page_start") or (min(pages) if pages else None),
        "page_end": get("page_end") or (max(pages) if pages else None),
        "pages": pages,
        "source_element_ids": list(get("source_element_ids", []) or []),
        "tags": list(metadata.get("tags") or get("tags", []) or []),
        "tag_provenance": dict(metadata.get("tag_provenance") or {}),
        "heading_path": list(metadata.get("heading_path") or get("heading_path", []) or []),
        "manifest_id": get("manifest_id"),
        "score": score,
        "evidence_role": role,
    }


def _stable_chunk_source_id(chunk: Any, embedding_model: str) -> str:
    chunk_id = str(getattr(chunk, "chunk_id", "") or "")
    file_hash = str(getattr(chunk, "file_hash", "") or "")
    from app.services.document_pipeline import stable_source_id
    generation = str(getattr(chunk, "generation", "") or "legacy")
    return stable_source_id(file_hash, generation, chunk_id)


async def _scoped_files(context: ToolInvocationContext, requested: str | None = None, services: DefaultToolServices | None = None) -> tuple[dict[str, Any], list[str]]:
    lookup = await (services or get_tool_services()).document_lookup(context.thread_id or "")
    if requested:
        if requested not in lookup:
            raise PermissionError("The requested document is not linked to this thread")
        return lookup, [requested]
    return lookup, list(lookup.keys())


async def search_knowledge(request: SearchKnowledgeRequest, context: ToolInvocationContext, *, services: DefaultToolServices | None = None):
    started = tool_started()
    tool_name = "search_knowledge"
    services = services or get_tool_services()
    try:
        if not context.thread_id or not context.embedding_model:
            return make_tool_result(tool_name=tool_name, content="No thread context found.", context=context, started=started, warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT])
        lookup, file_hashes = await _scoped_files(context, request.document_id, services)
        if not file_hashes:
            return make_tool_result(tool_name=tool_name, content="No documents are linked to this thread yet.", context=context, started=started, warnings=[ToolWarningCode.NO_THREAD_DOCUMENTS])
        repo = get_canonical_document_repo()
        from app.services.document_projection_service import evaluate_retrieval_readiness
        ready_manifests = []
        repair_scheduled = False
        from app.services.embedding_materialization_service import RESOURCE_DOCUMENT, ensure_embedding_job
        from app.services.document_pipeline import stable_fingerprint
        for file_hash in file_hashes:
            readiness = await evaluate_retrieval_readiness(file_hash, context.embedding_model)
            if readiness.get("ready") and readiness.get("manifest") is not None:
                ready_manifests.append(readiness["manifest"])
            else:
                await ensure_embedding_job(
                    resource_type=RESOURCE_DOCUMENT,
                    resource_id=file_hash,
                    scope_id=context.thread_id,
                    embedding_model=context.embedding_model,
                    source_version=readiness.get("repair_source_version") or stable_fingerprint("document-repair-v1", file_hash, context.embedding_model),
                    requeue_completed=True,
                )
                repair_scheduled = True
        if not ready_manifests:
            return make_tool_result(tool_name=tool_name, content="Document index is not ready for this thread.", context=context, started=started, warnings=[ToolWarningCode.MISSING_DOCUMENT_VECTORS, ToolWarningCode.INDEXING_IN_PROGRESS], artifacts={"readiness": "indexing", "repair_scheduled": repair_scheduled})
        ready_file_hashes = [manifest.file_hash for manifest in ready_manifests]
        repo = get_canonical_document_repo()
        scoped_section_ids: set[str] = set()
        if request.section_id:
            for manifest in ready_manifests:
                scoped_section_ids.update(
                    await repo.get_descendant_section_ids(
                        manifest.file_hash,
                        manifest.generation,
                        request.section_id,
                    )
                )
        query_vector = await services.embed(context.embedding_model, request.query)
        requested_pages = list(request.filters.pages) or _explicit_page_filter(request.query)
        candidate_limit = min(1000, max(100, request.max_results * 20))
        raw = await services.vector_db().search_knowledge_sources(
            thread_id=context.thread_id, query_vector=query_vector, embedding_model=context.embedding_model,
            limit=candidate_limit,
            file_hash=ready_file_hashes[0] if len(ready_file_hashes) == 1 else None,
            file_hashes=ready_file_hashes if len(ready_file_hashes) > 1 else None,
            query_text=request.query,
            pages=requested_pages,
            filters={
                "source_types": list(request.filters.source_types),
                "section_ids": sorted(scoped_section_ids),
                "tags": list(request.filters.tags),
                "manifest_ids": [manifest.manifest_id for manifest in ready_manifests],
                "generations": sorted({str(manifest.generation) for manifest in ready_manifests}),
            },
        )
        if not raw:
            return make_tool_result(tool_name=tool_name, content="No relevant content matched the requested document filters.", context=context, started=started, warnings=[ToolWarningCode.NO_RELEVANT_CONTENT], artifacts={"readiness": "ready", "repair_scheduled": repair_scheduled})
        if context.use_reranker and raw:
            raw = await services.rerank(request.query, raw)
        if request.section_id:
            raw = [
                item for item in raw
                if (item.get("metadata") or {}).get("section_id") in scoped_section_ids
                or item.get("section_id") in scoped_section_ids
            ]
        if request.filters.source_types:
            allowed_types = set(request.filters.source_types)
            raw = [item for item in raw if str(item.get("source_kind") or (item.get("metadata") or {}).get("source_kind") or "pdf") in allowed_types]
        if request.filters.tags:
            wanted_tags = set(request.filters.tags)
            raw = [item for item in raw if wanted_tags.intersection(set((item.get("metadata") or {}).get("tags") or item.get("tags") or []))]
        if requested_pages:
            wanted = set(requested_pages)
            def overlaps_requested_page(item: dict[str, Any]) -> bool:
                metadata = item.get("metadata") or {}
                page_start = item.get("page_start") or metadata.get("page_start")
                page_end = item.get("page_end") or metadata.get("page_end")
                if page_start is not None and page_end is not None:
                    return any(int(page_start) <= page <= int(page_end) for page in wanted)
                return bool(wanted.intersection(_normalise_pages(metadata.get("pages") or item.get("pages"))))
            raw = [item for item in raw if overlaps_requested_page(item)]
        if not raw:
            return make_tool_result(tool_name=tool_name, content="No relevant content matched the requested document filters.", context=context, started=started, warnings=[ToolWarningCode.NO_RELEVANT_CONTENT])

        if request.level == "chunk":
            matches = raw[:request.max_results]
        else:
            groups: dict[str, dict[str, Any]] = {}
            for item in raw:
                metadata = item.get("metadata") or {}
                file_hash = str(item.get("file_hash") or "")
                section_key = request.section_id if request.section_id else metadata.get("section_id") or file_hash
                key = file_hash if request.level == "document" else f"{file_hash}:{section_key}"
                if not file_hash or not section_key or not key:
                    continue
                group = groups.setdefault(str(key), {
                    "score": 0.0,
                    "file_hash": file_hash,
                    "section_id": metadata.get("section_id") if request.level == "section" else None,
                    "heading_path": metadata.get("heading_path") or [],
                    "pages": set(),
                    "source_element_ids": set(),
                    "text": item.get("text", ""),
                })
                group["score"] = max(float(group["score"] or 0), float(item.get("rerank_score", item.get("score", 0)) or 0))
                group["pages"].update(_normalise_pages(metadata.get("pages") or item.get("pages") or []))
                group["source_element_ids"].update(metadata.get("source_element_ids") or [])
            matches = sorted(groups.values(), key=lambda item: item["score"], reverse=True)[:request.max_results]

        sources: list[dict[str, Any]] = []
        content_parts: list[str] = []
        if request.level == "chunk":
            for item in matches:
                source = _source_from_chunk(item, score=item.get("rerank_score", item.get("score")))
                sources.append(source)
                metadata = item.get("metadata") or {}
                # Vector search stores contextualized_text in the generic
                # `text` property so embeddings retain heading context.  A
                # chunk-level answer, however, needs the original body text;
                # otherwise retrieval exposes only labels such as
                # "Document section" and cannot support grounded synthesis.
                body_text = str(metadata.get("body_text") or item.get("body_text") or "").strip()
                evidence_text = body_text or str(item.get("text") or "").strip()
                content_parts.append(f"[Source {source['source_id']} | pages {source.get('page_start') or '?'}]\n{evidence_text}")
        else:
            section_cache: dict[tuple[str, str], Any] = {}
            file_cache: dict[str, Any] = {}
            for item in matches:
                file_hash = str(item.get("file_hash") or "")
                section_id = request.section_id if request.level == "section" and request.section_id else item.get("section_id")
                if file_hash not in file_cache:
                    from app.db import get_file
                    file_cache[file_hash] = await get_file(file_hash)
                file_record = file_cache[file_hash]
                canonical = await repo.get(file_hash)
                source_metadata = dict(getattr(canonical, "source_metadata_json", None) or {}) if canonical else {}
                payload = getattr(canonical, "document_json", None) if canonical else {}
                title = (
                    getattr(file_record, "file_name", None)
                    or source_metadata.get("original_title")
                    or (payload or {}).get("filename")
                    or file_hash
                )
                if request.level == "section" and section_id:
                    cache_key = (file_hash, str(section_id))
                    sections = section_cache.get(cache_key)
                    if sections is None:
                        sections = await repo.get_sections(file_hash)
                        section_cache[cache_key] = sections
                    section = next((value for value in sections if value.section_id == section_id), None)
                    title = section.title if section else section_id
                source = {
                    "source_id": str(section_id if request.level == "section" and section_id else file_hash),
                    "file_hash": file_hash,
                    "parent_id": None,
                    "section_id": section_id if request.level == "section" else None,
                    "title": title,
                    "pages": _normalise_pages(item.get("pages") or []),
                    "source_element_ids": sorted(item.get("source_element_ids") or []),
                    "score": item.get("score"),
                    "evidence_role": "discovery",
                }
                sources.append(source)
                content_parts.append(f"[Discovery: {title}]\n{title}")
        content, truncated = bounded_retrieval_text(content_parts)
        artifacts = {
            "matches": sources,
            "document_sources": sources,
            "level": request.level,
            "readiness": "repair_in_progress" if repair_scheduled else "ready",
            "repair_scheduled": repair_scheduled,
            "truncated": truncated,
        }
        return make_tool_result(tool_name=tool_name, content=content, context=context, started=started, sources=sources, artifacts=artifacts, warnings=[ToolWarningCode.RESPONSE_TRUNCATED] if truncated else [])
    except PermissionError as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=str(exc), code="document_scope_forbidden", evidence_gap=True)
    except Exception as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=f"Error searching attached documents: {exc}", evidence_gap=True)


async def inspect_document(request: InspectDocumentRequest, context: ToolInvocationContext, *, services: DefaultToolServices | None = None):
    started = tool_started()
    tool_name = "inspect_document"
    try:
        if not context.thread_id:
            return make_tool_result(tool_name=tool_name, content="No thread context found.", context=context, started=started, warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT])
        await _scoped_files(context, request.document_id)
        repo = get_canonical_document_repo()
        from app.services.document_projection_service import evaluate_document_freshness
        freshness = await evaluate_document_freshness(request.document_id, context.embedding_model)
        canonical = freshness.get("canonical")
        if not freshness.get("canonical_ready"):
            if context.embedding_model:
                from app.services.embedding_materialization_service import RESOURCE_DOCUMENT, ensure_embedding_job
                await ensure_embedding_job(
                    resource_type=RESOURCE_DOCUMENT,
                    resource_id=request.document_id,
                    scope_id=context.thread_id,
                    embedding_model=context.embedding_model,
                    source_version=freshness.get("repair_source_version") or request.document_id,
                    requeue_completed=True,
                )
            return make_tool_result(tool_name=tool_name, content="Document conversion is not ready.", context=context, started=started, artifacts={"readiness": "conversion_in_progress", "repair_scheduled": bool(context.embedding_model)}, warnings=[ToolWarningCode.MISSING_DOCUMENT_VECTORS, ToolWarningCode.INDEXING_IN_PROGRESS])
        sections = await repo.get_sections(request.document_id, canonical.generation)
        if request.section_id:
            descendant_ids = set(await repo.get_descendant_section_ids(request.document_id, canonical.generation, request.section_id))
            sections = [item for item in sections if item.section_id in descendant_ids]
        offset = _cursor_offset(request.cursor)
        page = sections[offset:offset + request.page_size]
        entries = [{"section_id": item.section_id, "parent_id": item.parent_section_id, "title": item.title, "level": item.level, "heading_path": item.heading_path, "pages": [item.page_start, item.page_end], "expansion_targets": [item.section_id]} for item in page]
        elements = await repo.get_elements(request.document_id, canonical.generation)
        tables = sorted({str(item.element_id) for item in elements if item.element_type == "table"})
        next_cursor = _cursor(offset + len(page)) if offset + len(page) < len(sections) else None
        artifacts = {
            "outline": entries,
            "document_sources": entries,
            "tags": sorted({item.element_type for item in elements}),
            "tag_provenance": {item.element_type: "docling_label" for item in elements},
            "valid_expansion_targets": {
                "sections": [item.section_id for item in sections],
                "tables": tables,
            },
            "readiness": "ready",
            "continuation": next_cursor,
        }
        return make_tool_result(tool_name=tool_name, content=json.dumps(entries, ensure_ascii=False), context=context, started=started, artifacts=artifacts)
    except PermissionError as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=str(exc), code="document_scope_forbidden", evidence_gap=True)
    except Exception as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=f"Error inspecting document: {exc}")


async def read_context(request: ReadContextRequest, context: ToolInvocationContext, *, services: DefaultToolServices | None = None):
    started = tool_started()
    tool_name = "read_context"
    try:
        if not context.thread_id or not context.embedding_model:
            return make_tool_result(tool_name=tool_name, content="No thread context found.", context=context, started=started, warnings=[ToolWarningCode.MISSING_THREAD_CONTEXT])
        lookup, file_hashes = await _scoped_files(context, services=services)
        repo = get_canonical_document_repo()
        selected = []
        source_section_files: set[str] = set()
        source_anchors = []
        repair_scheduled = False
        for file_hash in file_hashes:
            from app.services.document_projection_service import evaluate_document_freshness
            freshness = await evaluate_document_freshness(
                file_hash,
                context.embedding_model,
                require_manifest=True,
            )
            manifest = freshness.get("manifest")
            if not freshness.get("canonical_ready") or not freshness.get("manifest_ready") or manifest is None:
                if context.embedding_model:
                    from app.services.embedding_materialization_service import RESOURCE_DOCUMENT, ensure_embedding_job
                    await ensure_embedding_job(
                        resource_type=RESOURCE_DOCUMENT,
                        resource_id=file_hash,
                        scope_id=context.thread_id,
                        embedding_model=context.embedding_model,
                        source_version=freshness.get("repair_source_version") or file_hash,
                        requeue_completed=True,
                    )
                    repair_scheduled = True
                continue
            canonical = await repo.get(file_hash)
            manifest_id = str(manifest.manifest_id)
            candidates = []
            if request.source_id.startswith("src_"):
                candidates = await repo.get_chunks_by_source_id(
                    request.source_id,
                    context.embedding_model,
                    file_hash=file_hash,
                    manifest_id=manifest_id,
                )
            if not candidates:
                candidates = await repo.get_chunks(
                    file_hash,
                    context.embedding_model,
                    manifest_id=manifest_id,
                )
            if canonical and canonical.status == "completed":
                section_ids = {
                    str(section.section_id)
                    for section in await repo.get_sections(file_hash, canonical.generation)
                }
                if request.source_id in section_ids:
                    source_section_files.add(file_hash)
                    selected.extend(candidates)
                    source_anchors.append(next(iter(candidates), None))
                    continue
            matches = [
                item for item in candidates
                if (
                    item.source_id == request.source_id
                    or item.chunk_id == request.source_id
                    or _stable_chunk_source_id(item, context.embedding_model) == request.source_id
                    or item.section_id == request.source_id
                    or item.table_id == request.source_id
                )
            ]
            selected.extend(matches)
            source_anchors.append(next(iter(matches), None))
        if not selected and repair_scheduled:
            return make_tool_result(tool_name=tool_name, content="Document context is being repaired.", context=context, started=started, warnings=[ToolWarningCode.MISSING_DOCUMENT_VECTORS, ToolWarningCode.INDEXING_IN_PROGRESS], artifacts={"readiness": "repair_in_progress", "repair_scheduled": True})
        if not selected:
            return make_tool_result(tool_name=tool_name, content="The requested source is not available in this thread.", context=context, started=started, warnings=[ToolWarningCode.NO_RELEVANT_CONTENT])
        source_chunk = next((item for item in source_anchors if item is not None), selected[0])
        canonical = await repo.get(source_chunk.file_hash)
        manifest_id = str(source_chunk.manifest_id)
        source_generation = canonical.generation if canonical and canonical.status == "completed" else ""
        source_is_section = bool(source_section_files) or any(item.section_id == request.source_id for item in selected)
        source_is_table = any(item.table_id == request.source_id for item in selected)
        if (request.expansion == "section" or (request.expansion == "chunk" and source_is_section)) and (source_is_section or source_chunk.section_id):
            section_id = request.source_id if source_is_section else source_chunk.section_id
            section_ids = set(
                await repo.get_descendant_section_ids(
                    source_chunk.file_hash,
                    source_generation,
                    section_id,
                )
            ) if source_is_section else {section_id}
            chunks = await repo.get_chunks(source_chunk.file_hash, context.embedding_model, manifest_id=manifest_id, section_ids=section_ids)
        elif (request.expansion == "table" or (request.expansion == "chunk" and source_is_table)) and source_chunk.table_id:
            table_id = request.source_id if source_is_table else source_chunk.table_id
            chunks = await repo.get_chunks(source_chunk.file_hash, context.embedding_model, manifest_id=manifest_id, table_id=table_id)
        else:
            chunks = [source_chunk]
        try:
            _tokenizer_config, counter = resolve_embedding_tokenizer(context.embedding_model)
        except EmbeddingTokenizerUnavailableError:
            raise
        token_budget = request.token_budget
        ordered_chunks = sorted(chunks, key=lambda item: int(item.chunk_order or 0))
        anchor_index = next((index for index, item in enumerate(ordered_chunks) if item.chunk_id == source_chunk.chunk_id), None)
        if anchor_index is None:
            raise ValueError("requested evidence is not present in the selected manifest")
        expansion_order = [ordered_chunks[anchor_index]]
        for distance in range(1, len(ordered_chunks)):
            for index in (anchor_index - distance, anchor_index + distance):
                if 0 <= index < len(ordered_chunks):
                    expansion_order.append(ordered_chunks[index])
        segments: list[tuple[Any, str]] = []
        for chunk in expansion_order:
            body = str(chunk.body_text or "").strip()
            if not body:
                continue
            segments.extend((chunk, segment) for segment in counter.split(body, token_budget))
        cursor = _context_cursor_payload(request.cursor)
        offset = int(cursor.get("segment_index", 0))
        if request.cursor and (
            str(cursor.get("manifest_id")) != manifest_id
            or str(cursor.get("anchor_chunk_id")) != str(source_chunk.chunk_id)
            or str(cursor.get("expansion")) != request.expansion
        ):
            raise ValueError("context continuation cursor does not match the requested evidence")
        total_segments = len(segments)
        segments = segments[offset:]
        parts: list[str] = []
        sources: list[dict[str, Any]] = []
        used = 0
        consumed = 0
        for index, (chunk, body) in enumerate(segments):
            candidate = "\n\n".join([*parts, body])
            if counter.count(candidate) > token_budget:
                break
            parts.append(body)
            used = counter.count(candidate)
            evidence = chunk.chunk_id == source_chunk.chunk_id or chunk.source_id == source_chunk.source_id
            sources.append(_source_from_chunk(chunk, role="evidence" if evidence else "surrounding_context"))
            consumed += 1
        content = "\n\n".join(parts)
        next_cursor = (
            _context_cursor(manifest_id, str(source_chunk.chunk_id), request.expansion, offset + consumed)
            if offset + consumed < total_segments else None
        )
        artifacts = {
            "document_sources": sources,
            "expansion": request.expansion,
            "readiness": "ready",
            "token_count": used,
            "truncated": next_cursor is not None,
            "continuation": next_cursor,
            "original_evidence_source_id": request.source_id,
        }
        warnings = [ToolWarningCode.RESPONSE_TRUNCATED] if artifacts["truncated"] else []
        return make_tool_result(tool_name=tool_name, content=content, context=context, started=started, sources=sources, artifacts=artifacts, warnings=warnings)
    except EmbeddingTokenizerUnavailableError as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=str(exc), code="embedding_tokenizer_unavailable", evidence_gap=True)
    except PermissionError as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=str(exc), code="document_scope_forbidden", evidence_gap=True)
    except Exception as exc:
        return make_tool_error_result(tool_name=tool_name, error=exc, context=context, started=started, user_message=f"Error reading document context: {exc}")


__all__ = ["inspect_document", "read_context", "search_knowledge"]
