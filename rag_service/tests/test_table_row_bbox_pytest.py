from app.services.document_conversion_service import (
    _align_coordinates,
    _split_shared_oversized_table_bboxes,
    _tighten_sentence_bbox_from_words,
)


def test_tighten_sentence_bbox_from_words_splits_oversized_line():
    sentence = {
        "page": 7,
        "text": "Transformer (big)",
        "bboxes": [{"page": 7, "x0": 10, "y0": 100, "x1": 362, "y1": 249}],
        "words": [
            {"x0": 12, "x1": 90, "top": 102, "bottom": 112, "page": 7},
            {"x0": 95, "x1": 130, "top": 102, "bottom": 112, "page": 7},
            {"x0": 12, "x1": 80, "top": 130, "bottom": 140, "page": 7},
        ],
    }
    _tighten_sentence_bbox_from_words(sentence)
    assert len(sentence["bboxes"]) == 2
    assert sentence["bboxes"][0]["y1"] - sentence["bboxes"][0]["y0"] < 40


def test_split_shared_oversized_table_bboxes_assigns_row_slices():
    sentences = [
        {
            "source_element_ids": ["tbl-1"],
            "page": 7,
            "text": "Row A",
            "bboxes": [{"page": 7, "x0": 10, "y0": 100, "x1": 362, "y1": 250}],
        },
        {
            "source_element_ids": ["tbl-1"],
            "page": 7,
            "text": "Row B",
            "bboxes": [{"page": 7, "x0": 10, "y0": 100, "x1": 362, "y1": 250}],
        },
    ]
    _split_shared_oversized_table_bboxes(sentences)
    assert sentences[0]["bboxes"][0]["y0"] == 100
    assert sentences[1]["bboxes"][0]["y0"] > sentences[0]["bboxes"][0]["y0"]
    assert sentences[1]["bboxes"][0]["y1"] <= 250


def test_align_coordinates_splits_shared_table_bbox_after_match():
    canonical = [
        {"text": "Alpha", "source_element_ids": ["tbl-2"], "page": 3},
        {"text": "Beta", "source_element_ids": ["tbl-2"], "page": 3},
    ]
    shared = {
        "page": 3,
        "x0": 0,
        "y0": 50,
        "x1": 300,
        "y1": 200,
        "bbox": {"page": 3, "x0": 0, "y0": 50, "x1": 300, "y1": 200},
        "bboxes": [{"page": 3, "x0": 0, "y0": 50, "x1": 300, "y1": 200}],
    }
    candidates = [
        {**shared, "text": "Alpha"},
        {**shared, "text": "Beta"},
    ]
    aligned = _align_coordinates(canonical, candidates)
    assert aligned[0]["bboxes"][0]["y0"] != aligned[1]["bboxes"][0]["y0"]
