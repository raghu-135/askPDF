"""Shared retrieval helpers for document and semantic history access."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.db import get_thread_shape
from app.models.llm_server_client import get_reranker_model, LOCAL_RERANKER_MODEL
from app.db.vector import get_vector_db

logger = logging.getLogger(__name__)
_DOCUMENT_VECTOR_TEMPORAL_FIELDS = {
    "document_available_in_thread_at",
    "document_indexed_at",
    "timeline_event_at",
    "timeline_event_type",
}


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
        merged["timeline_event_type"] = "document_added_to_thread"
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
        source_type = "pdf"
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
            group.get("source_type", "pdf"),
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
    embedding_model_name: str = None,
) -> Tuple[str, List[str]]:
    """Fetch semantic chat memory text plus the list of used message IDs."""

    db = get_vector_db()
    recalled = await db.search_chat_memory(
        thread_id=thread_id,
        query_vector=query_vector,
        embedding_model_name=embedding_model_name,
        limit=limit,
    )
    if use_reranker and query_text:
        recalled = await rerank_document_chunks(query_text, recalled)

    used_ids: List[str] = []
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

    return "\n\n---\n\n".join(parts), used_ids
