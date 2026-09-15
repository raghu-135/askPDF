"""Shared retrieval helpers for document and semantic history access."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.db import FileSourceType, get_thread_shape
from app.models.llm_server_client import get_reranker_model, LOCAL_RERANKER_MODEL
from app.db.vector import get_vector_db
from app.rag.enums import TimelineEventType
from runtime_protocol.tool_contract import MAX_TOOL_RESULT_STRING_LENGTH

logger = logging.getLogger(__name__)
RETRIEVAL_CONTENT_BUDGET = MAX_TOOL_RESULT_STRING_LENGTH - 2048
SEARCH_EVIDENCE_HIT_CHAR_LIMIT = 1500
_OMITTED_SOURCE_INSTRUCTION = (
    "Omitted source bodies due to size: {source_ids}. "
    "Call read_context with those source_id values and expansion=table when the source is tabular, "
    "otherwise expansion=section. Do not repeat search_knowledge with a similar query."
)
_DOCUMENT_VECTOR_TEMPORAL_FIELDS = {
    "document_available_in_thread_at",
    "document_indexed_at",
    "timeline_event_at",
    "timeline_event_type",
}


def bounded_retrieval_text(parts: List[str], *, budget: int = RETRIEVAL_CONTENT_BUDGET) -> Tuple[str, bool]:
    """Join complete evidence units without exceeding the shared content budget."""

    selected: List[str] = []
    used = 0
    truncated = False
    for part in parts:
        value = str(part or "")
        if not value:
            continue
        separator = 2 if selected else 0
        if used + separator + len(value) <= budget:
            selected.append(value)
            used += separator + len(value)
            continue
        truncated = True
        break
    return "\n\n".join(selected), truncated


def clip_retrieval_hit(value: str, *, hit_limit: int = SEARCH_EVIDENCE_HIT_CHAR_LIMIT) -> tuple[str, bool]:
    text = str(value or "")
    if len(text) <= hit_limit:
        return text, False
    return text[:hit_limit].rstrip() + "…", True


def bounded_retrieval_hits(
    parts: List[str],
    *,
    budget: int = RETRIEVAL_CONTENT_BUDGET,
    hit_limit: int = SEARCH_EVIDENCE_HIT_CHAR_LIMIT,
) -> tuple[list[tuple[int, str]], list[int], bool]:
    """Keep ranked hits that fit, clipping each body and recording omitted indexes."""

    selected: list[tuple[int, str]] = []
    used = 0
    truncated = False
    for index, part in enumerate(parts):
        value = str(part or "")
        if not value:
            continue
        clipped, clipped_hit = clip_retrieval_hit(value, hit_limit=hit_limit)
        separator = 2 if selected else 0
        if used + separator + len(clipped) <= budget:
            selected.append((index, clipped))
            used += separator + len(clipped)
            truncated = truncated or clipped_hit
            continue
        truncated = True
        omitted = [item for item in range(index, len(parts)) if str(parts[item] or "")]
        return selected, omitted, truncated
    return selected, [], truncated


def format_search_knowledge_content(
    sources: List[Dict[str, Any]],
    content_parts: List[str],
    *,
    budget: int = RETRIEVAL_CONTENT_BUDGET,
) -> tuple[str, bool]:
    """Build LLM-visible search text with a full source catalog and bounded bodies."""

    def catalog_line(index: int, status: str) -> str:
        source = sources[index] if index < len(sources) else {}
        bits: list[str] = []
        page_start = source.get("page_start")
        page_end = source.get("page_end")
        pages = source.get("pages") or []
        if page_start:
            bits.append(f"pages {page_start}-{page_end or page_start}")
        elif pages:
            bits.append(f"pages {min(pages)}-{max(pages)}")
        if source.get("table_id"):
            bits.append(f"table_id={source['table_id']}")
        section_id = source.get("section_id") or source.get("parent_id")
        if section_id:
            bits.append(f"section_id={section_id}")
        bits.append(status)
        extra = "; ".join(str(bit) for bit in bits if bit)
        source_id = source.get("source_id") or f"source-{index}"
        return f"- {source_id}" + (f" ({extra})" if extra else "")

    draft_catalog = "Ranked sources:\n" + "\n".join(catalog_line(index, "included") for index in range(len(sources)))
    hit_budget = max(1, budget - len(draft_catalog) - 480)
    included_indexes, omitted_indexes, truncated = bounded_retrieval_hits(content_parts, budget=hit_budget)
    included = {index for index, _body in included_indexes}
    catalog = "Ranked sources:\n" + "\n".join(
        catalog_line(index, "included" if index in included else "omitted")
        for index in range(len(sources))
    )
    bodies = "\n\n".join(body for _index, body in included_indexes)
    omitted_ids = [
        str(sources[index].get("source_id") or "")
        for index in omitted_indexes
        if index < len(sources) and sources[index].get("source_id")
    ]
    footer = _OMITTED_SOURCE_INSTRUCTION.format(source_ids=", ".join(omitted_ids)) if omitted_ids else ""
    if omitted_ids:
        truncated = True
    parts = [catalog]
    if bodies:
        parts.append(bodies)
    if footer:
        parts.append(footer)
    content = "\n\n".join(parts)
    if len(content) > budget:
        content = content[:budget].rstrip() + "…"
        truncated = True
    return content, truncated


async def get_document_name_lookup(thread_id: str) -> Dict[str, str]:
    """Return file_hash → file_name for all indexed documents in a thread."""

    try:
        shape = await get_thread_shape(thread_id)
        documents = shape.get("documents", {})
        # Filter out non-dict entries (e.g., 'updated_at' timestamp added by merge_jsonb_field)
        return {
            fh: meta.get("file_name", fh)
            for fh, meta in documents.items()
            if isinstance(meta, dict)
        }
    except Exception as exc:
        logger.warning("Failed to load thread document metadata: %s", exc)
        return {}


async def get_document_metadata_lookup(thread_id: str) -> Dict[str, Dict[str, Any]]:
    """Return file_hash → thread-local document inventory metadata."""

    try:
        shape = await get_thread_shape(thread_id)
        documents = shape.get("documents", {})
        return {
            fh: meta
            for fh, meta in documents.items()
            if isinstance(meta, dict)
        }
    except Exception as exc:
        logger.warning("Failed to load thread document metadata: %s", exc)
        return {}


def _merge_metadata(chunk: Dict[str, Any], thread_doc_meta: Dict[str, Any]) -> Dict[str, Any]:
    merged = {
        k: v
        for k, v in dict(chunk.get("metadata") or {}).items()
        if k not in _DOCUMENT_VECTOR_TEMPORAL_FIELDS
    }
    merged.update({
        k: v
        for k, v in chunk.items()
        if k not in ("metadata", "text") and k not in _DOCUMENT_VECTOR_TEMPORAL_FIELDS and v not in (None, "")
    })
    if thread_doc_meta.get("document_available_in_thread_at"):
        merged["document_available_in_thread_at"] = thread_doc_meta["document_available_in_thread_at"]
        merged["timeline_event_at"] = thread_doc_meta["document_available_in_thread_at"]
        merged["timeline_event_type"] = TimelineEventType.DOCUMENT_ADDED_TO_THREAD.value
    return merged


def _expand_pages(raw: Any) -> List[int]:
    pages: List[int] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            try:
                start = int(start_raw)
                end = int(end_raw)
            except Exception:
                continue
            if start <= end:
                pages.extend(range(start, end + 1))
            continue
        try:
            pages.append(int(part))
        except Exception:
            continue
    return pages


def _compact_page_ranges(raw_pages: List[Any]) -> str:
    pages: List[int] = []
    for raw in raw_pages:
        pages.extend(_expand_pages(raw))
    unique_pages = sorted(set(page for page in pages if page > 0))
    if not unique_pages:
        return ""

    ranges: List[str] = []
    start = prev = unique_pages[0]
    for page in unique_pages[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = page
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def _format_document_label(source_type: str, name: str, url: Optional[str], pages: Optional[str] = None) -> str:
    label_name = name or "Document"
    label = f"PDF: {label_name}"
    if pages:
        label = f"{label}, pages {pages}"
    return label


def group_document_chunks(
    chunks: List[Dict[str, Any]],
    hash_to_name: Optional[Dict[str, Any]] = None,
    char_budget: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Group document chunks and prepare context + source metadata."""

    doc_lookup = hash_to_name or {}
    document_sources: List[Dict[str, Any]] = []
    doc_groups: Dict[str, Dict[str, Any]] = {}
    used_chars = 0

    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue

        if char_budget and used_chars + len(text) > char_budget:
            break

        fh = chunk.get("file_hash") or ""
        source_type = FileSourceType.PDF.value
        url = chunk.get("url") or ""
        title = chunk.get("title") or ""
        raw_lookup = doc_lookup.get(fh, fh or "document")
        thread_doc_meta = raw_lookup if isinstance(raw_lookup, dict) else {}
        fallback_name = thread_doc_meta.get("file_name") or (raw_lookup if isinstance(raw_lookup, str) else fh or "document")
        name = title or fallback_name
        chunk_meta = _merge_metadata(chunk, thread_doc_meta)
        pages = chunk_meta.get("pages")

        if fh not in doc_groups:
            doc_groups[fh] = {
                "name": name,
                "source_type": source_type,
                "url": url,
                "texts": [],
                "pages": [],
                "document_available_in_thread_at": chunk_meta.get("document_available_in_thread_at"),
                "timeline_event_at": chunk_meta.get("timeline_event_at"),
                "timeline_event_type": chunk_meta.get("timeline_event_type"),
            }
        doc_groups[fh]["texts"].append(text)
        if pages:
            doc_groups[fh]["pages"].append(pages)

        used_chars += len(text)
        short_text = text if len(text) <= 200 else text[:200] + "..."
        score = chunk.get("rerank_score", chunk.get("score", 0.0))
        source_entry: Dict[str, Any] = {
            "text": short_text,
            "file_hash": chunk.get("file_hash"),
            "chunk_id": chunk.get("chunk_id"),
            "file_name": fallback_name,
            "title": title or None,
            "url": url or None,
            "source_type": source_type,
            "score": score,
        }
        for field in (
            "document_available_in_thread_at",
            "page_start",
            "page_end",
            "pages",
            "timeline_event_at",
            "timeline_event_type",
        ):
            value = chunk_meta.get(field)
            if value not in (None, ""):
                source_entry[field] = value
        document_sources.append(source_entry)

    context_parts: List[str] = []
    for group in doc_groups.values():
        combined_text = "\n".join(group["texts"])
        pages_label = _compact_page_ranges(group.get("pages", []))
        label = _format_document_label(
            group.get("source_type", FileSourceType.PDF.value),
            group.get("name", ""),
            group.get("url"),
            pages_label or None,
        )
        context_parts.append(f"[Source: {label}]\n{combined_text}")

    return "\n\n".join(context_parts), document_sources


async def rerank_document_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    if not chunks:
        return chunks

    reranker = get_reranker_model(model_name or LOCAL_RERANKER_MODEL)
    if reranker is None:
        return chunks

    passages = [c.get("text", "") for c in chunks]
    scores = await reranker.ascore(query, passages)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    ranked = sorted(chunks, key=lambda c: c.get("rerank_score", c.get("score", 0.0)), reverse=True)
    if top_k is not None:
        return ranked[:top_k]
    return ranked


async def fetch_semantic_history(
    thread_id: str,
    query_vector: List[float],
    query_text: Optional[str],
    limit: int,
    char_budget: Optional[int] = None,
    use_reranker: bool = True,
    embedding_model: str = None,
    include_refs: bool = False,
) -> tuple:
    """Fetch semantic chat memory text plus the list of used message IDs."""

    db = get_vector_db()
    recalled = await db.search_chat_memory(
        thread_id=thread_id,
        query_vector=query_vector,
        embedding_model=embedding_model,
        limit=limit,
    )
    if use_reranker and query_text:
        recalled = await rerank_document_chunks(query_text, recalled)

    used_ids: List[str] = []
    refs: List[Dict[str, Any]] = []
    parts: List[str] = []
    used_chars = 0

    for mem in recalled:
        text = mem.get("text", "")
        if not text:
            continue

        if char_budget and used_chars + len(text) > char_budget:
            break

        used_chars += len(text)
        message_created_at = mem.get("message_created_at")
        if message_created_at:
            parts.append(f"Earlier exchange at {message_created_at}:\n{text}")
        else:
            parts.append(text)
        if mem.get("message_id"):
            used_ids.append(mem["message_id"])
        ref: Dict[str, Any] = {
            key: mem.get(key)
            for key in (
                "message_id",
                "message_created_at",
                "score",
                "rerank_score",
            )
            if mem.get(key) not in (None, "")
        }
        if text:
            ref["preview"] = text if len(text) <= 260 else text[:260].rstrip() + "..."
            ref["content"] = text
        if ref:
            refs.append(ref)

    result = ("\n\n---\n\n".join(parts), used_ids)
    if include_refs:
        return result[0], result[1], refs
    return result
