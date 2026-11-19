import json
import os

import pytest

from app.db.jsonb_utils import SafeJSONB, postgres_jsonb_safe, postgres_text_safe


def test_postgres_jsonb_safe_roundtrips_nested_dict():
    value = {"text": "café", "n": 1}
    assert postgres_jsonb_safe(value) == value


def test_postgres_jsonb_safe_strips_lone_surrogates():
    dirty = {"text": "bad\ud800pair"}
    cleaned = postgres_jsonb_safe(dirty)
    assert "\ud800" not in cleaned["text"]
    assert postgres_jsonb_safe({"nested": [{"x": "\udc00"}]})["nested"][0]["x"]


def test_postgres_jsonb_safe_strips_nul_and_malformed_u_escapes():
    dirty = {
        "structure_json": {
            "raw": {"text": "Alu" + chr(92) + "u-containing mRNA\x00tail"},
            "heading_path": ["Introduction"],
        },
        "provenance_json": {"items": [{"note": "partial " + chr(92) + "u escape"}]},
    }
    cleaned = postgres_jsonb_safe(dirty)
    reencoded = json.dumps(cleaned)
    json.loads(reencoded)
    assert "\x00" not in cleaned["structure_json"]["raw"]["text"]


def test_postgres_text_safe_strips_nul_bytes():
    assert "\x00" not in postgres_text_safe("gene\x00name")


def test_safe_jsonb_type_decorator_scrubs_bind_params():
    safe_column = SafeJSONB()
    payload = {"text": "NUL\x00", "raw": chr(92) + "uXX"}
    bound = safe_column.process_bind_param(payload, dialect=None)
    assert bound is not None
    json.dumps(bound)


@pytest.mark.integration
def test_plos_biology_structure_json_survives_jsonb_scrub():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    pdf_path = os.path.join(
        repo_root,
        "benchmarks",
        "fixtures",
        "bundled",
        "plos-biology-0020391.pdf",
    )
    if not os.path.isfile(pdf_path):
        pytest.skip("bundled PLOS biology fixture not present")
    from app.services.document_pipeline import build_canonical_payload, derive_hierarchy
    from app.services.parsing_service import parse_with_docling

    with open(pdf_path, "rb") as handle:
        data = handle.read()
    parsed = parse_with_docling(data, "plos-biology-0020391.pdf")
    # Regression for asyncpg "unsupported Unicode escape sequence" on element JSONB.
    payload = build_canonical_payload(
        parsed.document,
        filename="plos-biology-0020391.pdf",
        source_metadata={"original_title": "plos-biology-0020391.pdf"},
        document_identity="biology-fixture",
    )
    _, elements = derive_hierarchy(payload)
    assert elements
    for element in elements[:50]:
        structure = {
            "raw": element.get("raw") or {},
            "heading_path": list(element.get("heading_path") or []),
            "table_structure": element.get("table_structure"),
        }
        provenance = {"items": list(element.get("provenance") or [])}
        payload = postgres_jsonb_safe(
            {"structure_json": structure, "provenance_json": provenance},
        )
        json.dumps(payload)
        json.loads(json.dumps(payload))