"""Deterministic Docling word-cell sentence alignment."""

from __future__ import annotations

from app.services.document_conversion_service import (
    _align_sentence_to_owned_cells,
    _align_sentences_to_cells,
    _assign_word_cells_to_elements,
    _group_cells_into_lines,
    _normalize_alignment_text,
)
from app.services.document_pipeline import (
    _collapsed_container_item,
    _merged_word_crumb_item,
)
from app.services.parsing_service import DoclingParseResult


def test_normalize_alignment_text_maps_unicode_and_strips_whitespace():
    raw = "hyph-\u00aden \u2018quote\u2019 \ufb01 lig"
    norm, index_map = _normalize_alignment_text(raw)
    assert norm == "hyph-en'quote'filig"
    assert index_map[0] == 0


def test_align_skips_cell_final_hyphen():
    element = {"text": "internationalization", "raw": {}}
    owned = [
        {"page": 1, "text": "inter-", "x0": 10, "x1": 30, "top": 100, "bottom": 110},
        {"page": 1, "text": "nationalization", "x0": 30, "x1": 120, "top": 100, "bottom": 110},
    ]
    sentence = {
        "text": "internationalization",
        "source_spans": [{"start": 0, "end": len("internationalization")}],
    }
    lines = _align_sentence_to_owned_cells(
        sentence,
        element,
        owned,
        docling_json={},
        page_sizes={1: (612.0, 792.0)},
    )
    assert lines is not None
    assert len(lines) == 1


def test_align_skips_list_markers_absent_in_element_text():
    element = {
        "text": "Item body only",
        "raw": {"_alignment_markers": ["1."]},
    }
    owned = [
        {"page": 1, "text": "1.", "x0": 10, "x1": 20, "top": 50, "bottom": 60},
        {"page": 1, "text": "Item", "x0": 22, "x1": 40, "top": 50, "bottom": 60},
        {"page": 1, "text": "body", "x0": 42, "x1": 70, "top": 50, "bottom": 60},
        {"page": 1, "text": "only", "x0": 72, "x1": 100, "top": 50, "bottom": 60},
    ]
    sentence = {
        "text": "Item body only",
        "source_spans": [{"start": 0, "end": len("Item body only")}],
    }
    lines = _align_sentence_to_owned_cells(
        sentence,
        element,
        owned,
        docling_json={},
        page_sizes={1: (612.0, 792.0)},
    )
    assert lines is not None


def test_mismatch_falls_back_to_coarse_via_align_sentences_to_cells():
    elements = [
        {
            "element_id": "el-1",
            "element_type": "text",
            "text": "Canonical sentence text.",
            "provenance": [
                {
                    "page_no": 1,
                    "bbox": {"l": 0, "t": 792, "r": 200, "b": 700, "coord_origin": "BOTTOMLEFT"},
                }
            ],
            "raw": {},
        }
    ]
    elements_by_id = {item["element_id"]: item for item in elements}
    parse_result = DoclingParseResult(
        document=None,
        page_sizes={1: (612.0, 792.0)},
        word_cells_by_page={
            1: [
                {"page": 1, "text": "Different", "x0": 10, "x1": 80, "top": 92, "bottom": 102},
                {"page": 1, "text": "words", "x0": 82, "x1": 120, "top": 92, "bottom": 102},
            ]
        },
    )
    sentences = [
        {
            "id": 0,
            "element_type": "text",
            "text": "Canonical sentence text.",
            "source_element_ids": ["el-1"],
            "source_spans": [{"start": 0, "end": len("Canonical sentence text.")}],
            "bboxes": [],
        }
    ]
    aligned = _align_sentences_to_cells(
        sentences,
        parse_result,
        elements_by_id=elements_by_id,
        docling_json={},
    )
    assert aligned[0]["alignment_precision"] == "coarse"
    assert aligned[0]["bboxes"]


def test_cell_ownership_uses_smallest_containing_provenance_box():
    elements = [
        {
            "element_id": "outer",
            "element_type": "text",
            "provenance": [
                {"page_no": 1, "bbox": {"l": 0, "t": 792, "r": 400, "b": 600, "coord_origin": "BOTTOMLEFT"}},
            ],
        },
        {
            "element_id": "inner",
            "element_type": "text",
            "provenance": [
                {"page_no": 1, "bbox": {"l": 50, "t": 692, "r": 150, "b": 682, "coord_origin": "BOTTOMLEFT"}},
            ],
        },
    ]
    cells = {
        1: [
            {"page": 1, "text": "tok", "x0": 80, "x1": 100, "top": 100, "bottom": 110},
        ]
    }
    owned = _assign_word_cells_to_elements(elements, cells, {1: (612.0, 792.0)})
    assert "inner" in owned
    assert owned["inner"][0]["text"] == "tok"
    assert "outer" not in owned or not owned["outer"]


def test_collapsed_container_keeps_child_provenance_boxes():
    collections = {
        "#/texts/0": {
            "label": "text",
            "text": "",
            "prov": [],
            "children": [{"$ref": "#/texts/1"}, {"$ref": "#/texts/2"}],
        },
        "#/texts/1": {
            "label": "text",
            "text": "Hello",
            "prov": [
                {"page_no": 1, "bbox": {"l": 10, "t": 700, "r": 50, "b": 680, "coord_origin": "BOTTOMLEFT"}},
            ],
        },
        "#/texts/2": {
            "label": "text",
            "text": "world",
            "prov": [
                {"page_no": 1, "bbox": {"l": 55, "t": 700, "r": 100, "b": 680, "coord_origin": "BOTTOMLEFT"}},
            ],
        },
    }
    collapsed = _collapsed_container_item(collections["#/texts/0"], ["#/texts/1", "#/texts/2"], collections)
    assert collapsed is not None
    assert collapsed["text"] == "Hello world"
    assert len(collapsed.get("prov") or []) == 2


def test_line_grouping_unions_same_line_cells():
    cells = [
        {"page": 1, "x0": 10, "x1": 30, "top": 100, "bottom": 110},
        {"page": 1, "x0": 32, "x1": 60, "top": 101, "bottom": 111},
        {"page": 1, "x0": 10, "x1": 40, "top": 120, "bottom": 130},
    ]
    lines = _group_cells_into_lines(cells, {1: (612.0, 792.0)})
    assert len(lines) == 2
    assert lines[0]["x1"] == 60
    assert lines[1]["y0"] == 120


def test_merged_word_crumb_collects_markers():
    collections = {
        "#/texts/0": {"label": "list_item", "text": "1.", "marker": "1.", "prov": [{"page_no": 1}]},
        "#/texts/1": {"label": "text", "text": "Intro", "prov": [{"page_no": 1}]},
    }
    merged = _merged_word_crumb_item(["#/texts/0", "#/texts/1"], collections)
    assert merged["text"] == "1. Intro"
    assert merged.get("_alignment_markers") == ["1."]
