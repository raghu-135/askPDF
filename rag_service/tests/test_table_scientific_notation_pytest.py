from app.services.document_pipeline import project_sentences, pack_retrieval_chunks, TokenCounter


def _counter() -> TokenCounter:
    def count(value: str) -> int:
        return len(value.replace("\n", " ").split())

    def split(value: str, limit: int):
        words = value.split()
        return [" ".join(words[i:i + limit]) for i in range(0, len(words), limit)]

    return TokenCounter(count=count, split=split)


def _table2_like_payload():
    # Minimal fixture capturing the split mantissa/exponent across EN-DE/EN-FR for base,
    # and a complete value with an empty sibling for big.
    return {
        "elements": [{
            "element_id": "tbl-2",
            "element_type": "table",
            "label": "table",
            "text": "Table 2 caption",
            "pages": [8],
            "table_structure": {
                "headers": [
                    "Model",
                    "BLEU/EN-DE",
                    "BLEU/EN-FR",
                    "Training Cost (FLOPs)/EN-DE",
                    "Training Cost (FLOPs)/EN-FR",
                ],
                "header_rows": [0, 1],
                "rows": [
                    ["Model", "BLEU", "", "Training Cost (FLOPs)", ""],
                    ["", "EN-DE", "EN-FR", "EN-DE", "EN-FR"],
                    ["Transformer (base model)", "27.3", "38.1", "3.3 ·", "1018"],
                    ["Transformer (big)", "28.4", "41.8", "2.3 · 1019", ""],
                ],
                "cells": [
                    {"row": 2, "col": 0, "row_span": 1, "col_span": 1, "text": "Transformer (base model)", "row_header": True},
                    {"row": 2, "col": 1, "row_span": 1, "col_span": 1, "text": "27.3"},
                    {"row": 2, "col": 2, "row_span": 1, "col_span": 1, "text": "38.1"},
                    {"row": 2, "col": 3, "row_span": 1, "col_span": 1, "text": "3.3 ·"},
                    {"row": 2, "col": 4, "row_span": 1, "col_span": 1, "text": "1018"},
                    {"row": 3, "col": 0, "row_span": 1, "col_span": 1, "text": "Transformer (big)", "row_header": True},
                    {"row": 3, "col": 1, "row_span": 1, "col_span": 1, "text": "28.4"},
                    {"row": 3, "col": 2, "row_span": 1, "col_span": 1, "text": "41.8"},
                    {"row": 3, "col": 3, "row_span": 1, "col_span": 1, "text": "2.3 · 1019"},
                ],
            },
        }]
    }


def test_scientific_notation_cells_are_coalesced_and_normalized():
    sentences = project_sentences(
        _table2_like_payload(),
        sentence_splitter=lambda value: [value],
        element_policy=None,
    )
    # Two data rows
    assert len(sentences) == 2
    base = sentences[0]
    big = sentences[1]
    # BLEU values preserved
    assert "BLEU/EN-DE: 27.3" in base["text"]
    assert "BLEU/EN-FR: 38.1" in base["text"]
    assert "BLEU/EN-DE: 28.4" in big["text"]
    assert "BLEU/EN-FR: 41.8" in big["text"]
    # Training costs merged across EN-DE/EN-FR and normalized
    assert "Training Cost (FLOPs)/EN-DE: 3.3 × 10^18" in base["text"]
    assert "Training Cost (FLOPs)/EN-FR: 3.3 × 10^18" in base["text"]
    assert "Training Cost (FLOPs)/EN-DE: 2.3 × 10^19" in big["text"]
    assert "Training Cost (FLOPs)/EN-FR: 2.3 × 10^19" in big["text"]
    # Row bodies carry the normalized values too
    assert base["table_row_body"].endswith("3.3 × 10^18")
    assert big["table_row_body"].endswith("2.3 × 10^19")


def test_each_table_row_forms_its_own_chunk():
    sentences = project_sentences(
        _table2_like_payload(),
        sentence_splitter=lambda value: [value],
        element_policy=None,
    )
    chunks = pack_retrieval_chunks(
        sentences,
        token_counter=_counter(),
        embedding_token_limit=256,
        document_identity="file:generation",
    )
    # One chunk per row
    assert len(chunks) == 2
    assert all(chunk.get("table_row_id") for chunk in chunks)
    assert chunks[0]["body_text"].count("Row 1:") == 1
    assert chunks[1]["body_text"].count("Row 2:") == 1
    assert "3.3 × 10^18" in chunks[0]["body_text"]
    assert "2.3 × 10^19" in chunks[1]["body_text"]
