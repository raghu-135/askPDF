"""Integration smoke: Docling word-cell alignment on sample PDF."""

from __future__ import annotations

import os

import pytest

from app.services.document_conversion_service import _align_sentences_to_cells
from app.services.document_pipeline import build_canonical_payload, derive_hierarchy, project_sentences
from app.services.parsing_service import parse_with_docling


@pytest.fixture
def sample_pdf_data():
    test_dir = os.path.dirname(__file__)
    with open(os.path.join(test_dir, "01030000000000.pdf"), "rb") as handle:
        return handle.read()


def test_sample_pdf_alignment_sets_exact_or_coarse(sample_pdf_data):
    filename = "01030000000000.pdf"
    parse_result = parse_with_docling(sample_pdf_data, filename)
    assert parse_result is not None
    payload = build_canonical_payload(
        parse_result.document,
        filename=filename,
        document_identity="alignment-coverage",
    )
    sections, elements = derive_hierarchy(payload)
    payload["sections"] = sections
    payload["elements"] = elements
    def _whole_text(value: str) -> list[str]:
        stripped = str(value or "").strip()
        return [stripped] if stripped else []

    sentences = project_sentences(payload, sentence_splitter=_whole_text)
    elements_by_id = {str(item["element_id"]): item for item in elements}
    aligned = _align_sentences_to_cells(
        sentences,
        parse_result,
        elements_by_id=elements_by_id,
        docling_json=payload.get("docling"),
    )
    non_table = [item for item in aligned if item.get("element_type") != "table"]
    assert non_table
    assert any(item.get("alignment_precision") in {"exact", "coarse"} for item in non_table)
