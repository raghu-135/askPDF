from app.agent.evidence_contract import (
    canonical_source_id,
    format_document_source_locator,
    parse_document_source_locator,
)


def test_format_document_source_locator_prefers_chunk_prefix():
    assert format_document_source_locator({"chunk_id": "src-1"}) == "c-src-1"
    assert format_document_source_locator({"sentence_id": 64}) == "s-64"


def test_canonical_source_id_uses_unambiguous_locators():
    file_hash = "a" * 32
    assert canonical_source_id({"file_hash": file_hash, "chunk_id": "src-1"}) == f"doc:{file_hash}:c-src-1"
    assert canonical_source_id({"file_hash": file_hash, "sentence_id": 12}) == f"doc:{file_hash}:s-12"


def test_parse_document_source_locator_never_crosses_prefixes():
    assert parse_document_source_locator("c-src-1") == ("chunk", "src-1")
    assert parse_document_source_locator("s-64") == ("sentence", "64")
    assert parse_document_source_locator("64") == ("legacy", "64")
