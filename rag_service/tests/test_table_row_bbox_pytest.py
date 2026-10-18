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


def _table_word(text: str, top: float, bottom: float) -> dict:
    return {
        "text": text,
        "x0": 129.5,
        "x1": 200.0,
        "top": top,
        "bottom": bottom,
        "page": 7,
        "page_width": 612.0,
        "page_height": 792.0,
    }


def test_split_shared_oversized_table_bboxes_uses_word_geometry_not_equal_slices():
    """Transformer (big) row must not inherit the base-model slice when Docling adds duplicate rows."""
    table_bbox = {"page": 7, "x0": 129.5, "y0": 95.0, "x1": 482.0, "y1": 244.3}
    word_pool = [
        _table_word("Transformer", 206.0, 218.0),
        _table_word("(base", 206.0, 218.0),
        _table_word("model)", 206.0, 218.0),
        _table_word("27.3", 206.0, 218.0),
        _table_word("Transformer", 219.0, 231.0),
        _table_word("(big)", 219.0, 231.0),
        _table_word("28.4", 219.0, 231.0),
    ]
    sentences = [
        {
            "source_element_ids": ["tbl-2"],
            "page": 7,
            "text": "Transformer (base model) 27.3",
            "bboxes": [dict(table_bbox)],
            "words": word_pool[:4],
        },
        {
            "source_element_ids": ["tbl-2"],
            "page": 7,
            "text": "Transformer (big) 28.4",
            "bboxes": [dict(table_bbox)],
            "words": word_pool[4:],
        },
    ]
    _split_shared_oversized_table_bboxes(sentences)
    base = sentences[0]["bboxes"][0]
    big = sentences[1]["bboxes"][0]
    assert abs(base["y0"] - 206.0) < 2
    assert abs(big["y0"] - 219.0) < 2
    assert big["y0"] > base["y0"]
    assert abs(big["y1"] - big["y0"] - 12.0) < 2


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
        {**shared, "text": "Alpha", "words": [{"text": "Alpha", "x0": 1, "x1": 40, "top": 55, "bottom": 65, "page": 3}]},
        {**shared, "text": "Beta", "words": [{"text": "Beta", "x0": 1, "x1": 40, "top": 80, "bottom": 90, "page": 3}]},
    ]
    aligned = _align_coordinates(canonical, candidates)
    assert aligned[0]["bboxes"][0]["y0"] != aligned[1]["bboxes"][0]["y0"]
