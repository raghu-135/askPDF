"""Reading-order reassembly after cross-encoder reranking."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.rag.retrieval import (
    _combine_group_bodies,
    expand_synthetic_sentence_siblings,
    group_document_chunks,
    reassemble_document_chunks_by_reading_order,
)


def _chunk(
    *,
    file_hash: str,
    section_id: str,
    chunk_index: int,
    text: str,
    score: float = 0.0,
    rerank_score: float | None = None,
    sentence_ids: list[str] | None = None,
    synthetic_span: bool = False,
    chunk_id: str | None = None,
) -> dict:
    metadata = {
        "body_text": text,
        "chunk_index": chunk_index,
        "section_id": section_id,
        "sentence_ids": sentence_ids or [str(chunk_index)],
    }
    if synthetic_span:
        metadata["synthetic_span"] = True
    return {
        "text": text,
        "file_hash": file_hash,
        "chunk_id": chunk_id or f"chunk-{chunk_index}",
        "source_id": chunk_id or f"chunk-{chunk_index}",
        "metadata": metadata,
        "score": score,
        **({"rerank_score": rerank_score} if rerank_score is not None else {}),
    }


def test_reassemble_section_chunks_in_reading_order_despite_rerank_scores():
    chunks = [
        _chunk(file_hash="f1", section_id="s1", chunk_index=0, text="first", rerank_score=0.2),
        _chunk(file_hash="f1", section_id="s1", chunk_index=1, text="second", rerank_score=0.9),
        _chunk(file_hash="f1", section_id="s1", chunk_index=2, text="third", rerank_score=0.3),
        _chunk(file_hash="f1", section_id="s2", chunk_index=0, text="other", rerank_score=0.95),
    ]
    groups = reassemble_document_chunks_by_reading_order(chunks)
    assert len(groups) == 2
    assert [item["metadata"]["chunk_index"] for item in groups[0]] == [0, 1, 2]
    assert " ".join(_combine_group_bodies(groups[0]).split()) == "first second third"
    assert groups[1][0]["metadata"]["section_id"] == "s2"


def test_group_document_chunks_emits_one_passage_per_section():
    chunks = [
        _chunk(file_hash="f1", section_id="s1", chunk_index=1, text="B", rerank_score=0.9),
        _chunk(file_hash="f1", section_id="s1", chunk_index=0, text="A", rerank_score=0.1),
    ]
    context, sources = group_document_chunks(chunks, {"f1": {"file_name": "doc.pdf"}})
    assert "A B" in context.replace("\n", " ")
    assert len(sources) == 2
    assert sources[0]["text"].startswith("A")
    assert sources[1]["text"].startswith("B")


@pytest.mark.asyncio
async def test_expand_synthetic_sentence_siblings_adds_missing_pieces(monkeypatch):
    chunks = [
        _chunk(
            file_hash="f1",
            section_id="s1",
            chunk_index=1,
            text="tail",
            sentence_ids=["7"],
            synthetic_span=True,
            chunk_id="c1",
            rerank_score=0.9,
        ),
    ]
    chunks[0]["manifest_id"] = "manifest-1"
    chunks[0]["metadata"]["manifest_id"] = "manifest-1"

    class FakeRepo:
        async def get_chunks(self, file_hash, embedding_model, *, manifest_id=None, section_id=None, **kwargs):
            return [
                SimpleNamespace(
                    chunk_id="c0",
                    file_hash=file_hash,
                    manifest_id=manifest_id,
                    section_id=section_id,
                    source_id="c0",
                    body_text="head",
                    chunk_order=0,
                    sentence_ids=["7"],
                    table_id=None,
                    metadata_json={"synthetic_span": True, "sentence_ids": ["7"]},
                ),
                SimpleNamespace(
                    chunk_id="c1",
                    file_hash=file_hash,
                    manifest_id=manifest_id,
                    section_id=section_id,
                    source_id="c1",
                    body_text="tail",
                    chunk_order=1,
                    sentence_ids=["7"],
                    table_id=None,
                    metadata_json={"synthetic_span": True, "sentence_ids": ["7"]},
                ),
            ]

    monkeypatch.setattr(
        "app.db.repositories.canonical_document_repo.get_canonical_document_repo",
        lambda: FakeRepo(),
    )
    expanded = await expand_synthetic_sentence_siblings(chunks, embedding_model="embed-1")
    bodies = sorted(
        item["metadata"]["chunk_index"] for item in expanded if item.get("file_hash") == "f1"
    )
    assert bodies == [0, 1]
