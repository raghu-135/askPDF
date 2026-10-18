"""Serve-time table row highlight boxes from Docling cells."""

import json
from pathlib import Path

from app.services.table_row_highlight_boxes import (
    _resolve_grid_row,
    _row_highlight_box,
    _table_bounds_topleft,
    apply_table_row_highlight_boxes,
)

_TABLE2_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "attention_table2_cells.json"


def _load_attention_table2_element() -> dict:
    table = json.loads(_TABLE2_FIXTURE.read_text(encoding="utf-8"))
    return {
        "element_id": table["element_id"],
        "label": "table",
        "pages": [8],
        "provenance": table["prov"],
        "raw": {
            "data": {
                "num_rows": table["num_rows"],
                "num_cols": table.get("num_cols", 5),
                "table_cells": [
                    {
                        **cell,
                        "start_row_offset_idx": cell["start_row_offset_idx"],
                        "end_row_offset_idx": cell["end_row_offset_idx"],
                        "start_col_offset_idx": cell["start_col_offset_idx"],
                        "end_col_offset_idx": cell["end_col_offset_idx"],
                    }
                    for cell in table["cells"]
                ],
            }
        },
    }


def _normalized_cells(element: dict) -> list[dict]:
    from app.services.table_row_highlight_boxes import _normalized_raw_cells

    return _normalized_raw_cells(element)


def test_row_highlight_box_ignores_vertical_span_for_row_spanning_cells():
    page_height = 792.0
    table_bounds = (100.0, 200.0, 500.0, 400.0)
    cells = [
        {
            "row_start": 7,
            "row_end": 10,
            "row_span": 3,
            "raw_cell": {
                "bbox": {
                    "l": 118.0,
                    "t": 238.0,
                    "r": 140.0,
                    "b": 301.0,
                    "coord_origin": "TOPLEFT",
                }
            },
        },
        {
            "row_start": 9,
            "row_end": 10,
            "row_span": 1,
            "raw_cell": {
                "bbox": {
                    "l": 236.0,
                    "t": 249.0,
                    "r": 457.0,
                    "b": 259.0,
                    "coord_origin": "TOPLEFT",
                }
            },
        },
    ]
    box = _row_highlight_box(
        cells,
        9,
        page=9,
        page_width=612.0,
        page_height=page_height,
        table_bounds=table_bounds,
    )
    assert box is not None
    assert box["y0"] == 249.0
    assert box["y1"] == 259.0
    assert box["x0"] == 100.0
    assert box["x1"] == 500.0


def test_row_highlight_box_clamps_x_to_table_edges():
    page_height = 792.0
    table_bounds = (136.7, 90.0, 475.3, 250.0)
    cells = [
        {
            "row_start": 10,
            "row_end": 11,
            "row_span": 1,
            "raw_cell": {
                "bbox": {
                    "l": 136.7,
                    "t": 218.4,
                    "r": 450.7,
                    "b": 228.8,
                    "coord_origin": "TOPLEFT",
                }
            },
        }
    ]
    box = _row_highlight_box(
        cells,
        10,
        page=8,
        page_width=612.0,
        page_height=page_height,
        table_bounds=table_bounds,
    )
    assert box["x0"] == 136.7
    assert box["x1"] == 475.3


def test_resolve_grid_row_uses_table_row_id_not_spanning_cell_row_start():
    element = {
        "element_id": "tbl-span",
        "table_structure": {
            "header_rows": [0],
            "rows": [
                ["Header"],
                ["(C)", "2"],
                ["", "4"],
                ["", "8"],
            ],
            "cells": [],
        },
    }
    cells = [
        {"index": 0, "row_start": 7, "row_end": 14, "row_span": 7, "raw_cell": {}},
        {"index": 1, "row_start": 9, "row_end": 10, "row_span": 1, "raw_cell": {}},
    ]
    sentence = {
        "table_row_id": "tbl-span:row:2",
        "table_cell_ids": ["tbl-span:cell:0", "tbl-span:cell:1"],
    }
    assert _resolve_grid_row(sentence, element, sentence["table_cell_ids"], cells) == 2


def test_spanning_group_rows_get_distinct_y_bands():
    page_height = 792.0
    table_bounds = (118.0, 130.0, 494.0, 380.0)
    cells = [
        {
            "row_start": 7,
            "row_end": 14,
            "row_span": 7,
            "raw_cell": {
                "bbox": {"l": 118.0, "t": 238.1, "r": 140.0, "b": 301.0, "coord_origin": "TOPLEFT"},
            },
        },
        {
            "row_start": 8,
            "row_end": 9,
            "row_span": 1,
            "raw_cell": {
                "bbox": {"l": 236.0, "t": 249.0, "r": 457.0, "b": 259.0, "coord_origin": "TOPLEFT"},
            },
        },
        {
            "row_start": 9,
            "row_end": 10,
            "row_span": 1,
            "raw_cell": {
                "bbox": {"l": 236.0, "t": 260.0, "r": 457.0, "b": 270.0, "coord_origin": "TOPLEFT"},
            },
        },
    ]
    row8 = _row_highlight_box(cells, 8, page=9, page_width=612.0, page_height=page_height, table_bounds=table_bounds)
    row9 = _row_highlight_box(cells, 9, page=9, page_width=612.0, page_height=page_height, table_bounds=table_bounds)
    assert row8 and row9
    assert abs(row8["y0"] - 249.0) < 0.6
    assert abs(row9["y0"] - 260.0) < 0.6
    assert row8["y0"] != row9["y0"]
    assert row8["x0"] == table_bounds[0]
    assert row8["x1"] == table_bounds[2]


def test_bytenet_row_uses_full_table_width():
    element = _load_attention_table2_element()
    element["table_structure"] = {
        "header_rows": [0, 1],
        "rows": [[], [], ["ByteNet"], []],
    }
    cells = _normalized_cells(element)
    bytenet_ids = [f"{element['element_id']}:cell:{cell['index']}" for cell in cells if cell["row_start"] == 2]
    sentence = {
        "table_row_id": f"{element['element_id']}:row:1",
        "table_cell_ids": bytenet_ids,
        "source_element_ids": [element["element_id"]],
        "page": 8,
        "pages": [8],
        "text": "Row 1: ByteNet",
        "bboxes": [],
    }
    page_height = 792.0
    table_bounds = _table_bounds_topleft(element, page_height)
    assert table_bounds is not None
    table_x0, _, table_x1, _ = table_bounds
    enriched = apply_table_row_highlight_boxes([sentence], {"elements": [element]})
    box = enriched[0]["bboxes"][0]
    assert abs(box["x0"] - table_x0) < 0.1
    assert abs(box["x1"] - table_x1) < 0.1


def test_attention_table2_transformer_rows_match_investigated_y_bands():
    element = _load_attention_table2_element()
    cells = _normalized_cells(element)
    table_bounds = (136.7, 99.0, 475.3, 244.3)
    page_height = 792.0
    base = _row_highlight_box(cells, 10, page=8, page_width=612.0, page_height=page_height, table_bounds=table_bounds)
    big = _row_highlight_box(cells, 11, page=8, page_width=612.0, page_height=page_height, table_bounds=table_bounds)
    assert base is not None and big is not None
    assert abs(base["y0"] - 218.4) < 0.6
    assert abs(base["y1"] - 228.8) < 0.6
    assert abs(big["y0"] - 229.8) < 0.6
    assert abs(big["y1"] - 240.2) < 0.6


def test_apply_table_row_highlight_boxes_preserves_sentence_text():
    element = _load_attention_table2_element()
    cells = _normalized_cells(element)
    base_ids = [
        f"{element['element_id']}:cell:{cell['index']}"
        for cell in cells
        if cell["row_start"] == 10
    ]
    sentences = [{
        "id": 227,
        "text": "Row 9: Transformer (base model) 27.3",
        "page": 8,
        "pages": [8],
        "source_element_ids": [element["element_id"]],
        "table_row_id": f"{element['element_id']}:row:9",
        "table_cell_ids": base_ids,
        "bboxes": [],
    }]
    document = {"elements": [element]}
    before_text = sentences[0]["text"]
    enriched = apply_table_row_highlight_boxes(sentences, document)
    assert enriched[0]["text"] == before_text
    assert enriched[0]["bboxes"]
    assert abs(enriched[0]["bboxes"][0]["y0"] - 218.4) < 0.6


def test_apply_falls_back_to_whole_table_when_row_geometry_missing():
    element = {
        "element_id": "tbl-x",
        "provenance": [{
            "page_no": 1,
            "bbox": {"l": 50.0, "t": 700.0, "r": 300.0, "b": 600.0, "coord_origin": "BOTTOMLEFT"},
        }],
        "raw": {"data": {"table_cells": []}},
    }
    sentences = [{
        "id": 1,
        "text": "Row 1: only text",
        "page": 1,
        "pages": [1],
        "source_element_ids": ["tbl-x"],
        "table_cell_ids": ["tbl-x:cell:0"],
        "bboxes": [],
    }]
    enriched = apply_table_row_highlight_boxes(sentences, {"elements": [element]})
    assert enriched[0]["bboxes"]
