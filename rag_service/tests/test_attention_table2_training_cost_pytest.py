from __future__ import annotations

import json
from pathlib import Path

from app.services.document_pipeline import project_sentences

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "attention_table2_cells.json"


def test_attention_table2_transformer_base_row_exposes_en_fr_training_cost():
    table = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    struct = table["table_structure"]
    payload = {
        "elements": [
            {
                "element_id": table["element_id"],
                "element_type": "table",
                "label": "table",
                "text": "Table 2",
                "pages": [8],
                "table_structure": {
                    "headers": [
                        "Model",
                        "BLEU/EN-DE",
                        "BLEU/EN-FR",
                        "Training Cost (FLOPs)/EN-DE",
                        "Training Cost (FLOPs)/EN-FR",
                    ],
                    "rows": struct["rows"],
                    "cells": table["cells"],
                },
            }
        ]
    }
    sentences = project_sentences(payload, sentence_splitter=lambda value: [value], element_policy=None)
    base_row = next(item for item in sentences if "Transformer (base model)" in item["text"])
    assert "Training Cost (FLOPs)/EN-FR: 3.3 × 10^18" in base_row["text"]
    assert "Training Cost (FLOPs)/EN-DE: 3.3 × 10^18" in base_row["text"]
    assert "1.0 × 10^20" not in base_row["text"]
