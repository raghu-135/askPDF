"""Coordinate alignment coverage on a small multi-column PDF fixture."""

from __future__ import annotations

import pytest

from app.services.document_conversion_service import (
    _align_coordinates,
    _normalize_match_text,
    _sentence_has_bbox,
)
from app.services.document_pipeline import build_canonical_payload, derive_hierarchy, project_sentences
from app.services.parsing_service import parse_with_docling, parse_with_pdfplumber


@pytest.fixture
def alignment_columns_pdf_bytes(tmp_path) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 72), "The ﬁrst column introduces attention mechanisms.", fontsize=10)
    page.insert_text((72, 96), "Line breaks can hyph-", fontsize=10)
    page.insert_text((72, 108), "enate words across rows.", fontsize=10)
    page.insert_text((320, 72), "The second column explains parallel layers.", fontsize=10)
    page.insert_text((320, 96), "Transformers remove recurrence entirely.", fontsize=10)
    pdf_path = tmp_path / "alignment_columns.pdf"
    doc.save(pdf_path)
    doc.close()
    return pdf_path.read_bytes()


def test_normalize_match_text_handles_ligatures_and_hyphenation():
    left = _normalize_match_text("The ﬁrst column hyph-\nenate")
    right = _normalize_match_text("The first column hyphenate")
    assert left == right


def test_spanning_alignment_joins_split_pdfplumber_sentences():
    canonical = [{"text": "Line breaks can hyphenate words across rows.", "pages": [1], "bboxes": []}]
    candidates = [
        {"text": "Line breaks can hyph-", "pages": [1], "bboxes": [{"page": 1, "x0": 1, "y0": 2, "x1": 3, "y1": 4}]},
        {"text": "enate words across rows.", "pages": [1], "bboxes": [{"page": 1, "x0": 5, "y0": 2, "x1": 8, "y1": 4}]},
    ]
    aligned = _align_coordinates(canonical, candidates)
    assert aligned[0]["alignment_precision"] == "exact"
    assert _sentence_has_bbox(aligned[0])


def test_provenance_fallback_assigns_non_zero_bbox():
    element = {
        "element_id": "el-1",
        "provenance": [
            {
                "page_no": 1,
                "bbox": {"l": 72.0, "t": 700.0, "r": 300.0, "b": 680.0},
            }
        ],
    }
    canonical = [
        {
            "text": "OCR-only sentence with no pdfplumber match.",
            "pages": [1],
            "source_element_ids": ["el-1"],
            "bboxes": [],
        }
    ]
    aligned = _align_coordinates(canonical, [], elements_by_id={"el-1": element})
    assert aligned[0]["alignment_precision"] == "coarse"
    assert _sentence_has_bbox(aligned[0])


@pytest.mark.integration
def test_alignment_bbox_coverage_on_multi_column_pdf(alignment_columns_pdf_bytes):
    try:
        from app.services.document_pipeline import _sentence_nlp

        _sentence_nlp()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"sentence pipeline unavailable: {exc}")
    data = alignment_columns_pdf_bytes
    filename = "alignment_columns.pdf"
    docling_doc = parse_with_docling(data, filename)
    assert docling_doc is not None
    payload = build_canonical_payload(docling_doc, filename=filename, document_identity="alignment-fixture")
    sections, elements = derive_hierarchy(payload)
    payload["sections"] = sections
    payload["elements"] = elements
    candidates = parse_with_pdfplumber(data, docling_doc, filename)
    sentences = project_sentences(payload)
    elements_by_id = {str(item["element_id"]): item for item in elements}
    aligned = _align_coordinates(
        sentences,
        candidates,
        elements_by_id=elements_by_id,
    )
    readable = [item for item in aligned if item.get("readable")]
    assert readable, "expected readable sentences from fixture PDF"
    with_bbox = [item for item in readable if _sentence_has_bbox(item)]
    coverage = len(with_bbox) / len(readable)
    assert coverage >= 0.9, f"expected >=90% bbox coverage, got {coverage:.1%} ({len(with_bbox)}/{len(readable)})"
