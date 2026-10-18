"""Guard: coordinate/highlight alignment must not change retrieval chunk inputs."""

from __future__ import annotations

import copy

import pytest

from app.services.document_conversion_service import (
    _align_coordinates,
    _merge_page_dimensions,
    _page_size_map_from_pdf,
)
from app.services.document_pipeline import (
    build_canonical_payload,
    derive_hierarchy,
    pack_retrieval_chunks,
    project_sentences,
)
from app.services.parsing_service import parse_with_docling


def _token_counter():
    def count(value: str) -> int:
        return len(value.replace("\n", " ").split())

    def split(value: str, limit: int):
        words = value.split()
        return [" ".join(words[i : i + limit]) for i in range(0, len(words), limit)]

    from app.services.document_pipeline import TokenCounter

    return TokenCounter(count=count, split=split)


def _retrieval_chunk_signature(chunks: list[dict]) -> list[dict]:
    return [
        {
            "sentence_ids": list(chunk["sentence_ids"]),
            "body_text": chunk["body_text"],
            "contextualized_text": chunk["contextualized_text"],
            "chunk_order": chunk["chunk_order"],
        }
        for chunk in chunks
    ]


def _stub_coordinate_candidates(
    sentences: list[dict],
    page_dimensions: dict[int, tuple[float, float]],
) -> list[dict]:
    """Synthetic pdfplumber rows: enough for alignment without spaCy/pdfplumber sentence NLP."""
    candidates: list[dict] = []
    y_cursor = 72.0
    for sentence in sentences:
        text = str(sentence.get("text") or "").strip()
        if not text:
            continue
        pages = [int(page) for page in (sentence.get("pages") or [1]) if int(page) > 0]
        page = pages[0] if pages else 1
        width, height = page_dimensions.get(page, (612.0, 792.0))
        bbox = {
            "page": page,
            "x0": 72.0,
            "y0": y_cursor,
            "x1": 320.0,
            "y1": y_cursor + 12.0,
            "page_width": width,
            "page_height": height,
        }
        candidates.append(
            {
                "text": text,
                "pages": pages or [page],
                "page": page,
                "label": sentence.get("label") or "text",
                "bbox": bbox,
                "bboxes": [bbox],
                "page_width": width,
                "page_height": height,
            }
        )
        y_cursor += 14.0
    return candidates


def _sentence_split_whole(text: str) -> list[str]:
    stripped = str(text or "").strip()
    return [stripped] if stripped else []


def _chunks_for_fixture_pdf(data: bytes, filename: str, *, align_coordinates: bool) -> list[dict]:
    docling_doc = parse_with_docling(data, filename)
    assert docling_doc is not None
    payload = build_canonical_payload(docling_doc, filename=filename, document_identity="highlight-invariant-fixture")
    sections, elements = derive_hierarchy(payload)
    payload["sections"] = sections
    payload["elements"] = elements
    sentences = project_sentences(payload, sentence_splitter=_sentence_split_whole)
    if align_coordinates:
        elements_by_id = {str(item["element_id"]): item for item in elements}
        page_dimensions = _merge_page_dimensions(_page_size_map_from_pdf(data), {})
        candidates = _stub_coordinate_candidates(sentences, page_dimensions)
        sentences = _align_coordinates(
            copy.deepcopy(sentences),
            candidates,
            elements_by_id=elements_by_id,
            page_dimensions=page_dimensions,
        )
    return pack_retrieval_chunks(
        sentences,
        token_counter=_token_counter(),
        embedding_token_limit=128,
        document_identity="highlight-invariant-fixture",
        document_title=filename,
    )


@pytest.fixture
def highlight_invariant_pdf_bytes(tmp_path) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "The first column introduces attention mechanisms.", fontsize=10)
    page.insert_text((72, 96), "Line breaks can hyph-", fontsize=10)
    page.insert_text((72, 108), "enate words across rows.", fontsize=10)
    page.insert_text((320, 72), "The second column explains parallel layers.", fontsize=10)
    page.insert_text((320, 96), "Transformers remove recurrence entirely.", fontsize=10)
    pdf_path = tmp_path / "highlight_invariant.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path.read_bytes()


def test_coordinate_alignment_does_not_change_retrieval_chunks(highlight_invariant_pdf_bytes):
    data = highlight_invariant_pdf_bytes
    filename = "highlight_invariant.pdf"
    without_alignment = _chunks_for_fixture_pdf(data, filename, align_coordinates=False)
    with_alignment = _chunks_for_fixture_pdf(data, filename, align_coordinates=True)
    assert without_alignment, "expected chunks from fixture PDF"
    assert _retrieval_chunk_signature(without_alignment) == _retrieval_chunk_signature(with_alignment)
