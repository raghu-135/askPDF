from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import or_, select

from app.agent.evidence_contract import (
    RECORD_IDENTIFIER_PATTERN,
    SOURCE_IDENTIFIER_PATTERN,
    format_document_source_locator,
    parse_document_source_locator,
    normalized_canonical_source_id,
    normalized_source_url,
)
from app.db.connection_sqlmodel import async_session_maker
from app.db.models_sqlmodel import DocumentChunk
from app.db import get_file_parsed_sentences
from app.db.repositories.canonical_document_repo import get_canonical_document_repo
from app.models.canvas import (
    CanvasCitation,
    CanvasSection,
    CanvasSpec,
    ClaimPairBlock,
    ClaimSide,
    MarkdownBlock,
    QuoteBlock,
    SourceListBlock,
    StatBlock,
    TableBlock,
    TableCell,
    TimelineBlock,
    TimelineEvent,
)
from app.models.canvas_draft import CanvasCompileError, CanvasDraftClaim, CanvasPublishDraft

_FILE_HASH_RE = re.compile(r"^[0-9a-f]{32}$", re.IGNORECASE)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ResolvedClaim:
    text: str
    citations: list[CanvasCitation]
    quote: Optional[tuple[str, int]]  # file_hash, sentence_id
    tone: Optional[str]
    time: Optional[str]
    group: Optional[str]


async def _chunk_rows_for_table_id(file_hash: str, table_id: str) -> list[Any]:
    if not table_id:
        return []
    async with async_session_maker() as session:
        async with session.begin():
            result = await session.execute(
                select(DocumentChunk)
                .where(
                    DocumentChunk.file_hash == file_hash,
                    DocumentChunk.table_id == table_id,
                )
                .order_by(DocumentChunk.chunk_order)
            )
            return list(result.scalars().all())


async def _chunk_rows_ordered(file_hash: str) -> list[Any]:
    async with async_session_maker() as session:
        async with session.begin():
            result = await session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.file_hash == file_hash)
                .order_by(DocumentChunk.chunk_order)
            )
            return list(result.scalars().all())


async def _chunk_row_by_order_index(file_hash: str, index: int) -> Any | None:
    if index < 0:
        return None
    rows = await _chunk_rows_ordered(file_hash)
    if index >= len(rows):
        return None
    return rows[index]


async def _repair_stale_chunk_locator(
    file_hash: str,
    locator: str,
    *,
    claim_text: str,
    embedding_model: str | None = None,
) -> str | None:
    """Remap invented ordinal locators (e.g. c-0) to real chunk ids for this file."""
    kind, token = parse_document_source_locator(locator)
    if kind != "chunk":
        return None
    if await _chunk_locator_exists(file_hash, token):
        return token
    candidate_row: Any | None = None
    if token.isdigit():
        candidate_row = await _chunk_row_by_order_index(file_hash, int(token))
    if candidate_row is None:
        rows = await _chunk_rows_ordered(file_hash)
        if not rows:
            return None
        claim = str(claim_text or "").strip()
        if claim:
            by_id = await _parsed_sentence_catalog(file_hash)
            best_score = 0.0
            for row in rows:
                row_token = parse_document_source_locator(
                    format_document_source_locator({"chunk_id": row.chunk_id or row.source_id}),
                )[1]
                scroll_id, _highlight, found = await _readable_sentence_ids_for_chunk(
                    file_hash,
                    row_token,
                    embedding_model=embedding_model,
                    claim_text=claim,
                )
                if not found or scroll_id is None:
                    continue
                item = by_id.get(scroll_id) or {}
                score = _sentence_claim_overlap_score(claim, str(item.get("text") or ""))
                if score > best_score:
                    best_score = score
                    candidate_row = row
        if candidate_row is None:
            candidate_row = rows[0]
    repaired_locator = format_document_source_locator({"chunk_id": candidate_row.chunk_id or candidate_row.source_id})
    repaired_token = parse_document_source_locator(repaired_locator)[1]
    if not repaired_token or not await _chunk_locator_exists(file_hash, repaired_token):
        return None
    if repaired_token != token:
        logger.info(
            "canvas compile repaired stale locator doc:%s:%s -> doc:%s:%s",
            file_hash,
            locator,
            file_hash,
            repaired_locator,
        )
    return repaired_token


async def _chunk_row_for_locator(
    file_hash: str,
    locator: str,
    *,
    embedding_model: str | None = None,
):
    async with async_session_maker() as session:
        async with session.begin():
            result = await session.execute(
                select(DocumentChunk)
                .where(
                    DocumentChunk.file_hash == file_hash,
                    or_(
                        DocumentChunk.chunk_id == locator,
                        DocumentChunk.source_id == locator,
                        DocumentChunk.section_id == locator,
                        DocumentChunk.table_id == locator,
                    ),
                )
                .order_by(DocumentChunk.chunk_order)
                .limit(1)
            )
            row = result.scalar_one_or_none()
    if row is not None:
        return row
    if not embedding_model:
        return None
    repo = get_canonical_document_repo()
    chunks = await repo.get_chunks_by_source_id(locator, embedding_model, file_hash=file_hash)
    return chunks[0] if chunks else None


async def _chunk_locator_exists(file_hash: str, locator: str) -> bool:
    return await _chunk_row_for_locator(file_hash, locator) is not None


async def _parsed_sentence_catalog(file_hash: str) -> dict[int, dict[str, Any]]:
    parsed = await get_file_parsed_sentences(file_hash)
    by_id: dict[int, dict[str, Any]] = {}
    for item in (parsed or {}).get("sentences") or []:
        if not isinstance(item, dict):
            continue
        try:
            by_id[int(item.get("id"))] = item
        except (TypeError, ValueError):
            continue
    return by_id


def _claim_match_tokens(claim_text: str) -> set[str]:
    tokens = set(re.findall(r"[a-z0-9]+", str(claim_text or "").casefold()))
    return {token for token in tokens if token.isdigit() or len(token) >= 3}


def _sentence_claim_overlap_score(claim_text: str, sentence_text: str) -> float:
    claim_tokens = _claim_match_tokens(claim_text)
    if not claim_tokens:
        return 0.0
    lowered = str(sentence_text or "").casefold()
    claim_numbers = {token for token in claim_tokens if token.isdigit()}
    sentence_numbers = set(re.findall(r"\d+", lowered))
    score = 0.0
    if claim_numbers and sentence_numbers:
        score += 3.0 * len(claim_numbers & sentence_numbers)
    for token in claim_tokens:
        if token.isdigit():
            continue
        if token in lowered:
            score += 1.0
    return score


def _catalog_ids_from_chunk_rows(
    rows: list[Any],
    by_id: dict[int, dict[str, Any]],
) -> tuple[list[int], list[int]]:
    catalog_ids: list[int] = []
    readable_ids: list[int] = []
    seen: set[int] = set()
    for row in rows:
        for raw in row.sentence_ids or []:
            try:
                sentence_id = int(raw)
            except (TypeError, ValueError):
                continue
            if sentence_id in seen:
                continue
            item = by_id.get(sentence_id)
            if not item or not str(item.get("text") or "").strip():
                continue
            seen.add(sentence_id)
            catalog_ids.append(sentence_id)
            if item.get("readable") is not False:
                readable_ids.append(sentence_id)
    return catalog_ids, readable_ids


def _sentence_ids_from_chunk_row(
    row: Any,
    by_id: dict[int, dict[str, Any]],
    *,
    claim_text: str = "",
) -> tuple[int | None, list[int]]:
    """Return scroll target (first readable) and highlight ids for a chunk row."""
    catalog_ids, readable_ids = _catalog_ids_from_chunk_rows([row], by_id)
    claim = str(claim_text or "").strip()
    if claim and catalog_ids:
        scored: list[tuple[float, int, bool]] = []
        for sentence_id in catalog_ids:
            item = by_id[sentence_id]
            score = _sentence_claim_overlap_score(claim, str(item.get("text") or ""))
            if score > 0:
                scored.append((score, sentence_id, item.get("readable") is not False))
        if scored:
            scored.sort(key=lambda entry: (-entry[0], entry[1]))
            highlight_ids = [sentence_id for _, sentence_id, _ in scored]
            readable_matches = [sentence_id for _, sentence_id, readable in scored if readable]
            scroll_id = (readable_matches or highlight_ids)[0]
            return scroll_id, highlight_ids
    highlight_ids = readable_ids or catalog_ids
    scroll_id = readable_ids[0] if readable_ids else (catalog_ids[0] if catalog_ids else None)
    return scroll_id, highlight_ids


async def _readable_sentence_ids_for_chunk(
    file_hash: str,
    locator: str,
    *,
    embedding_model: str | None = None,
    claim_text: str = "",
) -> tuple[int | None, list[int], bool]:
    """Resolve chunk locator to sentence ids. Third value is True when a chunk row was found."""
    row = await _chunk_row_for_locator(file_hash, locator, embedding_model=embedding_model)
    if row is None:
        return None, [], False
    by_id = await _parsed_sentence_catalog(file_hash)
    table_id = str(getattr(row, "table_id", None) or "").strip()
    rows = [row]
    if table_id:
        siblings = await _chunk_rows_for_table_id(file_hash, table_id)
        if siblings:
            rows = siblings
    if len(rows) > 1:
        catalog_ids, readable_ids = _catalog_ids_from_chunk_rows(rows, by_id)
        claim = str(claim_text or "").strip()
        if claim and catalog_ids:
            scored: list[tuple[float, int, bool]] = []
            for sentence_id in catalog_ids:
                item = by_id[sentence_id]
                score = _sentence_claim_overlap_score(claim, str(item.get("text") or ""))
                if score > 0:
                    scored.append((score, sentence_id, item.get("readable") is not False))
            if scored:
                scored.sort(key=lambda entry: (-entry[0], entry[1]))
                highlight_ids = [sentence_id for _, sentence_id, _ in scored]
                readable_matches = [sentence_id for _, sentence_id, readable in scored if readable]
                scroll_id = (readable_matches or highlight_ids)[0]
                return scroll_id, highlight_ids, True
        highlight_ids = readable_ids or catalog_ids
        scroll_id = readable_ids[0] if readable_ids else (catalog_ids[0] if catalog_ids else None)
        return scroll_id, highlight_ids, True
    scroll_id, highlight_ids = _sentence_ids_from_chunk_row(row, by_id, claim_text=claim_text)
    return scroll_id, highlight_ids, True


async def _first_sentence_for_chunk(file_hash: str, locator: str) -> int | None:
    first, _readable, _found = await _readable_sentence_ids_for_chunk(file_hash, locator)
    return first


def _raise_unresolved_document_locator(file_hash: str, locator: str, *, reason: str) -> None:
    message = f"could not resolve doc:{file_hash}:{locator} ({reason})"
    logger.warning("canvas compile unresolved document locator: %s", message)
    raise CanvasCompileError("unknown_chunk", message)


async def _sentence_exists(file_hash: str, sentence_id: int) -> bool:
    parsed = await get_file_parsed_sentences(file_hash)
    if not parsed:
        return False
    for item in parsed.get("sentences") or []:
        if not isinstance(item, dict):
            continue
        try:
            if int(item.get("id")) == sentence_id:
                return bool(str(item.get("text") or "").strip())
        except (TypeError, ValueError):
            continue
    return False


async def _canonicalize_draft_source_id(
    source_id: str,
    *,
    thread_file_hashes: set[str],
) -> str:
    normalized = normalized_canonical_source_id(source_id)
    if normalized:
        return normalized
    raw = str(source_id or "").strip()
    if not raw:
        return ""
    async with async_session_maker() as session:
        async with session.begin():
            if raw.startswith("src_"):
                row = (
                    await session.execute(
                        select(DocumentChunk)
                        .where(DocumentChunk.source_id == raw)
                        .order_by(DocumentChunk.chunk_order)
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if row is not None and row.file_hash in thread_file_hashes:
                    return f"doc:{row.file_hash}:{format_document_source_locator({'chunk_id': row.chunk_id or row.source_id})}"
            row = (
                await session.execute(
                    select(DocumentChunk)
                    .where(
                        or_(DocumentChunk.section_id == raw, DocumentChunk.table_id == raw),
                    )
                    .order_by(DocumentChunk.chunk_order)
                    .limit(1)
                )
            ).scalar_one_or_none()
            if row is not None and row.file_hash in thread_file_hashes:
                return f"doc:{row.file_hash}:{format_document_source_locator({'chunk_id': row.chunk_id or row.source_id})}"
    return ""


async def _citation_for_document_locator(
    file_hash: str,
    locator: str,
    *,
    default_label: str,
    embedding_model: str | None = None,
) -> tuple[CanvasCitation, Optional[tuple[str, int]]]:
    quote: Optional[tuple[str, int]] = None
    sentence_ids: list[int] = []
    kind, token = parse_document_source_locator(locator)
    if kind == "sentence":
        try:
            sentence_id = int(token)
        except ValueError:
            raise CanvasCompileError("unknown_sentence", f"sentence locator is invalid for doc:{file_hash}:{locator}")
        if not await _sentence_exists(file_hash, sentence_id):
            raise CanvasCompileError(
                "unknown_sentence",
                f"sentence_id {sentence_id} was not found for doc:{file_hash}:{locator}",
            )
        quote = (file_hash, sentence_id)
        sentence_ids = [sentence_id]
    elif kind == "legacy":
        legacy_id = token
        sentence_exists = await _sentence_exists(file_hash, int(legacy_id))
        chunk_exists = await _chunk_locator_exists(file_hash, legacy_id)
        if sentence_exists and not chunk_exists:
            sentence_id = int(legacy_id)
            quote = (file_hash, sentence_id)
            sentence_ids = [sentence_id]
        else:
            first, readable, found = await _readable_sentence_ids_for_chunk(
                file_hash,
                legacy_id,
                embedding_model=embedding_model,
                claim_text=default_label,
            )
            if not found:
                _raise_unresolved_document_locator(file_hash, locator, reason="chunk row not found for legacy locator")
            if not readable:
                _raise_unresolved_document_locator(
                    file_hash,
                    locator,
                    reason="chunk has no sentence ids in the current reading catalog",
                )
            quote = (file_hash, first) if first is not None else None
            sentence_ids = readable
    else:
        chunk_token = token
        first, readable, found = await _readable_sentence_ids_for_chunk(
            file_hash,
            chunk_token,
            embedding_model=embedding_model,
            claim_text=default_label,
        )
        if not found:
            repaired_token = await _repair_stale_chunk_locator(
                file_hash,
                locator,
                claim_text=default_label,
                embedding_model=embedding_model,
            )
            if repaired_token:
                chunk_token = repaired_token
                first, readable, found = await _readable_sentence_ids_for_chunk(
                    file_hash,
                    chunk_token,
                    embedding_model=embedding_model,
                    claim_text=default_label,
                )
        if not found:
            _raise_unresolved_document_locator(file_hash, locator, reason="chunk row not found (stale c- locator or re-chunked document)")
        if not readable:
            _raise_unresolved_document_locator(
                file_hash,
                locator,
                reason="chunk sentence_ids do not match the current sentence catalog",
            )
        if first is not None and await _sentence_exists(file_hash, first):
            quote = (file_hash, first)
        sentence_ids = readable
    if not sentence_ids:
        _raise_unresolved_document_locator(file_hash, locator, reason="no highlightable sentence ids")
    scroll_id = sentence_ids[0]
    citation = CanvasCitation(
        kind="document",
        label=default_label[:160],
        file_hash=file_hash,
        sentence_id=scroll_id,
        sentence_ids=sentence_ids,
    )
    if quote is None:
        quote = (file_hash, scroll_id)
    return citation, quote


async def _resolve_src_alias(
    source_id: str,
    *,
    thread_file_hashes: set[str],
    embedding_model: str | None,
    default_label: str,
) -> tuple[CanvasCitation, Optional[tuple[str, int]]] | None:
    if not str(source_id or "").startswith("src_"):
        return None
    if not embedding_model:
        raise CanvasCompileError(
            "invalid_source_id",
            f"cannot resolve retrieval source_id {source_id} without thread embedding model",
        )
    repo = get_canonical_document_repo()
    for file_hash in sorted(thread_file_hashes):
        chunks = await repo.get_chunks_by_source_id(source_id, embedding_model, file_hash=file_hash)
        if not chunks:
            continue
        chunk = chunks[0]
        locator = format_document_source_locator({"chunk_id": chunk.chunk_id or chunk.source_id})
        if not locator:
            continue
        return await _citation_for_document_locator(
            file_hash,
            locator,
            default_label=default_label,
            embedding_model=embedding_model,
        )
    return None


async def _resolve_source_id(
    source_id: str,
    *,
    thread_file_hashes: set[str],
    embedding_model: str | None,
    default_label: str,
) -> tuple[CanvasCitation, Optional[tuple[str, int]]]:
    src_resolved = await _resolve_src_alias(
        source_id,
        thread_file_hashes=thread_file_hashes,
        embedding_model=embedding_model,
        default_label=default_label,
    )
    if src_resolved is not None:
        return src_resolved

    canonical = await _canonicalize_draft_source_id(source_id, thread_file_hashes=thread_file_hashes)
    normalized = canonical or normalized_canonical_source_id(source_id)
    if not normalized:
        raise CanvasCompileError("invalid_source_id", f"source_id is not valid: {source_id}")

    if normalized.startswith("web:"):
        url = normalized_source_url(normalized[4:])
        if not url:
            raise CanvasCompileError("invalid_source_id", f"web source_id has no url: {source_id}")
        return CanvasCitation(kind="web", label=default_label[:160], url=url), None

    if normalized.startswith("conversation:"):
        message_id = normalized[len("conversation:") :]
        if not RECORD_IDENTIFIER_PATTERN.fullmatch(message_id):
            raise CanvasCompileError("invalid_source_id", f"conversation source_id is invalid: {source_id}")
        return CanvasCitation(kind="conversation", label=default_label[:160], message_id=message_id), None

    if normalized.startswith("memory:"):
        memory_id = normalized[len("memory:") :]
        if not RECORD_IDENTIFIER_PATTERN.fullmatch(memory_id):
            raise CanvasCompileError("invalid_source_id", f"memory source_id is invalid: {source_id}")
        return CanvasCitation(kind="memory", label=default_label[:160], memory_id=memory_id), None

    if not normalized.startswith("doc:"):
        raise CanvasCompileError("invalid_source_id", f"unsupported source_id prefix: {source_id}")

    parts = normalized.split(":", 2)
    if len(parts) != 3:
        raise CanvasCompileError("invalid_source_id", f"document source_id is malformed: {source_id}")
    file_hash, locator = parts[1], parts[2]
    if not _FILE_HASH_RE.fullmatch(file_hash) or not SOURCE_IDENTIFIER_PATTERN.fullmatch(file_hash):
        raise CanvasCompileError("unknown_document", f"document file_hash is invalid in {source_id}")
    if file_hash not in thread_file_hashes:
        raise CanvasCompileError(
            "unknown_document",
            f"document {file_hash} is not attached to this thread (source_id {source_id})",
        )

    return await _citation_for_document_locator(
        file_hash,
        locator,
        default_label=default_label,
        embedding_model=embedding_model,
    )


async def resolve_draft_claims(
    claims: list[CanvasDraftClaim],
    *,
    thread_file_hashes: set[str],
    embedding_model: str | None = None,
) -> list[ResolvedClaim]:
    resolved: list[ResolvedClaim] = []
    for claim in claims:
        citations: list[CanvasCitation] = []
        quote: Optional[tuple[str, int]] = None
        label = claim.text.strip()[:160] or "Source"
        for source_id in claim.source_ids:
            try:
                citation, claim_quote = await _resolve_source_id(
                    source_id,
                    thread_file_hashes=thread_file_hashes,
                    embedding_model=embedding_model,
                    default_label=label,
                )
            except CanvasCompileError as exc:
                if exc.code not in {"unknown_chunk", "unknown_sentence"}:
                    raise
                logger.warning("canvas compile dropping unresolvable source_id %s: %s", source_id, exc)
                continue
            citations.append(citation)
            if claim_quote and quote is None:
                quote = claim_quote
        if not citations:
            raise CanvasCompileError("missing_citations", "each claim needs at least one resolvable source_id")
        resolved.append(
            ResolvedClaim(
                text=claim.text.strip(),
                citations=citations,
                quote=quote,
                tone=claim.tone,
                time=claim.time.strip() if claim.time else None,
                group=claim.group.strip() if claim.group else None,
            )
        )
    return resolved


def _unique_citations(claims: list[ResolvedClaim]) -> list[CanvasCitation]:
    seen: set[tuple[Any, ...]] = set()
    ordered: list[CanvasCitation] = []
    for claim in claims:
        for citation in claim.citations:
            key = (
                citation.kind,
                citation.file_hash,
                citation.sentence_id,
                tuple(citation.sentence_ids or ()),
                citation.url,
                citation.memory_id,
                citation.message_id,
                citation.label,
            )
            if key in seen:
                continue
            seen.add(key)
            ordered.append(citation)
    return ordered


def _stat_blocks(claims: list[ResolvedClaim], layout: str) -> list[StatBlock]:
    if layout == "timeline":
        times = [c.time for c in claims if c.time]
        span = f"{times[0]} – {times[-1]}" if len(times) >= 2 else (times[0] if times else "n/a")
        return [
            StatBlock(value=str(len(claims)), label="Events", tone="info"),
            StatBlock(value=span, label="Span", tone="neutral"),
        ]
    groups = {c.group for c in claims if c.group}
    supported = sum(1 for c in claims if c.tone == "supported")
    contradicted = sum(1 for c in claims if c.tone == "contradicted")
    stats = [
        StatBlock(value=str(len(claims)), label="Claims reviewed", tone="info"),
    ]
    if layout == "matrix":
        stats.append(StatBlock(value=str(supported), label="Supported", tone="neutral"))
        stats.append(StatBlock(value=str(contradicted), label="Contradicted", tone="warning" if contradicted else "neutral"))
    elif layout == "compare" and groups:
        stats.append(StatBlock(value=str(len(groups)), label="Sources compared", tone="neutral"))
    return stats[:4]


def _compile_compare(claims: list[ResolvedClaim]) -> list[Any]:
    groups = sorted({c.group or "Source" for c in claims})
    if len(groups) < 2:
        groups = groups or ["Claim"]
    max_cols = min(7, len(groups))
    headers = ["Claim"] + groups[:max_cols]
    rows: list[list[TableCell]] = []
    for claim in claims:
        row: list[TableCell] = [TableCell(text=claim.text[:400])]
        target = claim.group or groups[0]
        for group in headers[1:]:
            if group == target:
                row.append(TableCell(text=claim.text[:400]))
            else:
                row.append(TableCell(text=""))
        rows.append(row)

    blocks: list[Any] = [
        TableBlock(caption="Comparison", headers=headers, rows=rows),
    ]
    contradicted = [c for c in claims if c.tone == "contradicted"]
    if len(contradicted) >= 2:
        left, right = contradicted[0], contradicted[1]
        blocks.append(
            ClaimPairBlock(
                relation="contradicts",
                left=ClaimSide(text=left.text, citation=left.citations[0]),
                right=ClaimSide(text=right.text, citation=right.citations[0]),
            )
        )
    elif len(claims) >= 2:
        blocks.append(
            ClaimPairBlock(
                relation="extends",
                left=ClaimSide(text=claims[0].text, citation=claims[0].citations[0]),
                right=ClaimSide(text=claims[1].text, citation=claims[1].citations[0]),
            )
        )
    return blocks


def _compile_matrix(claims: list[ResolvedClaim]) -> list[Any]:
    headers = ["Claim", "Source", "Support", "Notes"]
    rows: list[list[TableCell]] = []
    for claim in claims:
        citation = claim.citations[0]
        support_text = claim.tone or "partial"
        cell = TableCell(text=support_text, tone=claim.tone) if claim.tone else TableCell(text=support_text)
        rows.append([
            TableCell(text=claim.text[:400]),
            TableCell(text=citation.label[:400]),
            cell,
            TableCell(text=""),
        ])
    return [TableBlock(caption="Evidence matrix", headers=headers, rows=rows)]


def _compile_timeline(claims: list[ResolvedClaim]) -> list[Any]:
    events = [
        TimelineEvent(
            time=claim.time or "—",
            label=claim.text[:200],
            citation=claim.citations[0],
        )
        for claim in claims
    ]
    return [TimelineBlock(title="Timeline", events=events)]


def _quote_blocks(claims: list[ResolvedClaim]) -> list[QuoteBlock]:
    blocks: list[QuoteBlock] = []
    seen: set[tuple[str, int]] = set()
    for claim in claims:
        if not claim.quote:
            continue
        key = claim.quote
        if key in seen:
            continue
        seen.add(key)
        blocks.append(
            QuoteBlock(
                file_hash=key[0],
                sentence_id=key[1],
                label=claim.text[:160],
            )
        )
    return blocks[:3]


async def compile_canvas_draft(
    draft: CanvasPublishDraft,
    *,
    thread_file_hashes: set[str],
    embedding_model: str | None = None,
) -> CanvasSpec:
    claims = await resolve_draft_claims(
        draft.claims,
        thread_file_hashes=thread_file_hashes,
        embedding_model=embedding_model,
    )
    citations = _unique_citations(claims)
    stats = _stat_blocks(claims, draft.layout)
    if draft.layout == "compare":
        body_blocks = _compile_compare(claims)
        section_title = "Comparison"
    elif draft.layout == "matrix":
        body_blocks = _compile_matrix(claims)
        section_title = "Evidence matrix"
    else:
        body_blocks = _compile_timeline(claims)
        section_title = "Timeline"

    quote_blocks = _quote_blocks(claims)
    overview_blocks: list[Any] = list(stats)
    if draft.summary:
        overview_blocks.append(MarkdownBlock(text=draft.summary.strip()))

    sections = [
        CanvasSection(title="Overview", blocks=overview_blocks),
        CanvasSection(title=section_title, blocks=[*body_blocks, *quote_blocks]),
        CanvasSection(
            title="Sources",
            blocks=[SourceListBlock(citations=citations)],
        ),
    ]
    return CanvasSpec(
        schema_version=1,
        title=draft.title.strip(),
        summary=draft.summary.strip() if draft.summary else None,
        sections=sections,
    )
