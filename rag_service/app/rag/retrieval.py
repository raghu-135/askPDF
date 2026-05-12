"""Shared retrieval helpers for document and semantic history access."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.db import get_thread_shape
from app.models.llm_server_client import get_reranker_model, LOCAL_RERANKER_MODEL
from app.db.vector import get_vector_db

logger = logging.getLogger(__name__)


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


def _format_document_label(source_type: str, name: str, url: Optional[str]) -> str:
    label_name = name or "Document"
    return f"PDF: {label_name}"


def group_document_chunks(
    chunks: List[Dict[str, Any]],
    hash_to_name: Optional[Dict[str, str]] = None,
    char_budget: Optional[int] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Group document chunks and prepare context + source metadata."""

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
        source_type = "pdf"
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
        score = chunk.get("rerank_score", chunk.get("score", 0.0))
        document_sources.append({
            "text": short_text,
            "file_hash": chunk.get("file_hash"),
            "file_name": fallback_name,
            "title": title or None,
            "url": url or None,
            "source_type": source_type,
            "score": score,
        })

    context_parts: List[str] = []
    for group in doc_groups.values():
        combined_text = "\n".join(group["texts"])
        label = _format_document_label(group.get("source_type", "pdf"), group.get("name", ""), group.get("url"))
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
        parts.append(text)
        if mem.get("message_id"):
            used_ids.append(mem["message_id"])

    return "\n\n---\n\n".join(parts), used_ids
