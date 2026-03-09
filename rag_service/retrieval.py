"""Shared retrieval helpers for document and semantic history access."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from database import get_thread_shape
from vectordb.qdrant import get_qdrant

logger = logging.getLogger(__name__)


async def get_document_name_lookup(thread_id: str) -> Dict[str, str]:
    """Return file_hash → file_name for all indexed documents in a thread."""

    try:
        shape = await get_thread_shape(thread_id)
        return {fh: meta.get("file_name", fh) for fh, meta in shape.get("documents", {}).items()}
    except Exception as exc:
        logger.warning("Failed to load thread document metadata: %s", exc)
        return {}


def _format_document_label(source_type: str, name: str, url: Optional[str]) -> str:
    stype = (source_type or "document").lower()
    if stype in {"webpage", "web"}:
        title = name or "Webpage"
        if url:
            return f"Webpage: {title} | {url}"
        return f"Webpage: {title}"
    # Default to PDF-style label
    label_name = name or "Document"
    return f"PDF: {label_name}"


def group_document_chunks(
    chunks: List[Dict[str, Any]],
    hash_to_name: Optional[Dict[str, str]] = None,
    char_budget: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Group document chunks (PDFs + webpages) and prepare context + source metadata."""

    hash_to_name = hash_to_name or {}
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
        source_type = chunk.get("source_kind") or chunk.get("source_type") or "pdf"
        if source_type == "web":
            source_type = "webpage"
        url = chunk.get("url") or ""
        title = chunk.get("title") or ""
        fallback_name = hash_to_name.get(fh, fh or "document")
        name = title or fallback_name

        if fh not in doc_groups:
            doc_groups[fh] = {
                "name": name,
                "source_type": source_type,
                "url": url,
                "texts": [],
            }
        doc_groups[fh]["texts"].append(text)

        used_chars += len(text)
        short_text = text if len(text) <= 200 else text[:200] + "..."
        document_sources.append({
            "text": short_text,
            "file_hash": chunk.get("file_hash"),
            "file_name": fallback_name,
            "title": title or None,
            "url": url or None,
            "source_type": source_type,
            "score": chunk.get("score", 0.0),
        })

    context_parts: List[str] = []
    for group in doc_groups.values():
        combined_text = "\n".join(group["texts"])
        label = _format_document_label(group.get("source_type", "pdf"), group.get("name", ""), group.get("url"))
        context_parts.append(f"[Source: {label}]\n{combined_text}")

    return "\n\n".join(context_parts), document_sources


async def fetch_semantic_history(
    thread_id: str,
    query_vector: List[float],
    limit: int,
    char_budget: Optional[int] = None,
) -> Tuple[str, List[str]]:
    """Fetch semantic chat memory text plus the list of used message IDs."""

    db = get_qdrant()
    recalled = await db.search_chat_memory(
        thread_id=thread_id,
        query_vector=query_vector,
        limit=limit,
    )

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
        parts.append(text)
        if mem.get("message_id"):
            used_ids.append(mem["message_id"])

    return "\n\n---\n\n".join(parts), used_ids
