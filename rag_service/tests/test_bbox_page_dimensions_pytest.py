from app.services.document_conversion_service import (
    _align_coordinates,
    _apply_page_dimensions_to_sentences,
    _bbox_within_page,
    _bboxes_from_element_provenance,
)


def test_provenance_bboxes_include_page_dimensions_and_stay_inside_page():
    page_dimensions = {1: (612.0, 792.0)}
    element = {
        "element_id": "el-1",
        "provenance": [
            {
                "page_no": 1,
                "bbox": {"l": 72.0, "t": 700.0, "r": 300.0, "b": 680.0},
            }
        ],
    }
    bboxes = _bboxes_from_element_provenance(element, page_dimensions=page_dimensions)
    assert len(bboxes) == 1
    box = bboxes[0]
    assert box["page_width"] == 612.0
    assert box["page_height"] == 792.0
    assert _bbox_within_page(box, box["page_width"], box["page_height"])


def test_apply_page_dimensions_stamps_aligned_sentence_boxes():
    page_dimensions = {1: (400.0, 600.0)}
    sentences = [
        {
            "page": 1,
            "bboxes": [{"page": 1, "x0": 10, "y0": 20, "x1": 90, "y1": 40}],
            "bbox": {"page": 1, "x0": 10, "y0": 20, "x1": 90, "y1": 40},
        }
    ]
    _apply_page_dimensions_to_sentences(sentences, page_dimensions)
    box = sentences[0]["bboxes"][0]
    assert box["page_width"] == 400.0
    assert box["page_height"] == 600.0
    assert sentences[0]["page_width"] == 400.0
    assert sentences[0]["page_height"] == 600.0


def test_align_coordinates_stamps_page_sizes_on_provenance_fallback():
    page_dimensions = {1: (612.0, 792.0)}
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
    aligned = _align_coordinates(
        canonical,
        [],
        elements_by_id={"el-1": element},
        page_dimensions=page_dimensions,
    )
    box = aligned[0]["bboxes"][0]
    assert box["page_width"] > 0
    assert box["page_height"] > 0
    assert _bbox_within_page(box, box["page_width"], box["page_height"])
