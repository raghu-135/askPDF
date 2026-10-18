"""Table row sentence projection must align grid indices with rendered text."""

from app.services.document_pipeline import project_sentences


def _two_row_header_table_payload():
    return {
        "elements": [{
            "element_id": "tbl-bleU",
            "element_type": "table",
            "label": "table",
            "text": "Table 2 caption",
            "pages": [8],
            "table_structure": {
                "headers": ["Model", "BLEU", "Cost"],
                "header_rows": [0, 1],
                "rows": [
                    ["Model", "BLEU", "Cost"],
                    ["", "EN-DE", "EN-FR"],
                    ["ByteNet [18]", "23.75", ""],
                    ["Deep-Att + PosUnk [39]", "39.2", "1.0e20"],
                    ["Transformer (big)", "28.4", "41.8"],
                ],
                "cells": [
                    {"row": 2, "col": 0, "row_span": 1, "col_span": 1, "text": "ByteNet [18]"},
                    {"row": 2, "col": 1, "row_span": 1, "col_span": 1, "text": "23.75"},
                    {"row": 3, "col": 0, "row_span": 1, "col_span": 1, "text": "Deep-Att + PosUnk [39]"},
                    {"row": 3, "col": 1, "row_span": 1, "col_span": 1, "text": "39.2"},
                    {"row": 3, "col": 2, "row_span": 1, "col_span": 1, "text": "1.0e20"},
                    {"row": 4, "col": 0, "row_span": 1, "col_span": 1, "text": "Transformer (big)"},
                    {"row": 4, "col": 1, "row_span": 1, "col_span": 1, "text": "28.4"},
                    {"row": 4, "col": 2, "row_span": 1, "col_span": 1, "text": "41.8"},
                ],
            },
        }]
    }


def test_project_sentences_skips_header_rows_and_matches_grid_cells():
    sentences = project_sentences(
        _two_row_header_table_payload(),
        sentence_splitter=lambda value: [value],
        element_policy=None,
    )
    assert len(sentences) == 3
    assert sentences[0]["text"].startswith("Row 1:")
    assert "ByteNet" in sentences[0]["text"]
    assert "23.75" in sentences[0]["text"]
    assert "39.2" not in sentences[0]["text"]
    assert sentences[1]["text"].startswith("Row 2:")
    assert "Deep-Att" in sentences[1]["text"]
    assert "39.2" in sentences[1]["text"]
    assert sentences[2]["text"].startswith("Row 3:")
    assert "Transformer (big)" in sentences[2]["text"]
    assert "28.4" in sentences[2]["text"]
    assert sentences[0]["table_row_id"] == "tbl-bleU:row:1"
    assert any(cell.endswith(":cell:0") for cell in sentences[0]["table_cell_ids"])
    assert not any(cell.endswith(":cell:2") for cell in sentences[0]["table_cell_ids"])


def test_project_sentences_does_not_emit_phantom_row_beyond_grid():
    payload = _two_row_header_table_payload()
    payload["elements"][0]["table_structure"]["rows"].append(["", "", ""])
    sentences = project_sentences(
        payload,
        sentence_splitter=lambda value: [value],
        element_policy=None,
    )
    assert len(sentences) == 3
    assert all("Transformer (big)" in sentence["text"] for sentence in sentences if "Row 3" in sentence["text"])
