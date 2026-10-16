import pytest

from app.services.document_pipeline import (
    TokenCounter,
    build_canonical_payload,
    derive_hierarchy,
    pack_retrieval_chunks,
    project_sentences,
    stable_source_id,
    CANONICAL_SCHEMA_VERSION,
)
from app.services.embedding_tokenizer import EmbeddingTokenizerConfig


def _counter() -> TokenCounter:
    def count(value: str) -> int:
        return len(value.replace("\n", " ").split())

    def split(value: str, limit: int):
        words = value.split()
        return [" ".join(words[i:i + limit]) for i in range(0, len(words), limit)]

    return TokenCounter(count=count, split=split)


def test_pack_fills_embedding_budget_and_preserves_context_and_ids():
    sentences = [
        {"id": 1, "text": "one", "section_id": "s1", "heading_path": ["Root", "Child"], "source_element_ids": ["e1"], "pages": [2]},
        {"id": 2, "text": "two", "section_id": "s1", "heading_path": ["Root", "Child"], "source_element_ids": ["e2"], "pages": [2]},
        {"id": 3, "text": "three", "section_id": "s1", "heading_path": ["Root", "Child"], "source_element_ids": ["e3"], "pages": [3]},
        {"id": 4, "text": "four", "section_id": "s1", "heading_path": ["Root", "Child"], "source_element_ids": ["e4"], "pages": [3]},
        {"id": 5, "text": "五 unicode", "section_id": "s2", "heading_path": ["Other"], "source_element_ids": ["e5"], "pages": [4]},
    ]
    chunks = pack_retrieval_chunks(sentences, token_counter=_counter(), embedding_token_limit=16)
    assert chunks[0]["sentence_ids"] == ["1", "2", "3", "4"]
    assert chunks[-1]["section_id"] == "s2"
    assert all(chunk["token_count"] <= 16 for chunk in chunks)
    assert "Child" in chunks[0]["contextualized_text"]
    assert "one two three" in chunks[0]["body_text"]
    assert "four" in " ".join(chunk["body_text"] for chunk in chunks if chunk["section_id"] == "s1")


def test_pack_keeps_table_boundaries_while_grouping_text_elements():
    sentences = [
        {"id": 1, "text": "first", "section_id": "s", "source_element_ids": ["paragraph-1"]},
        {"id": 2, "text": "second", "section_id": "s", "source_element_ids": ["paragraph-2"]},
        {"id": 3, "text": "header", "section_id": "s", "table_id": "table-1", "source_element_ids": ["table-1"]},
        {"id": 4, "text": "row", "section_id": "s", "table_id": "table-1", "source_element_ids": ["table-1"]},
    ]
    chunks = pack_retrieval_chunks(sentences, token_counter=_counter(), embedding_token_limit=32)
    assert [chunk["sentence_ids"] for chunk in chunks] == [["1", "2"], ["3", "4"]]


def test_long_sentence_splits_deterministically_but_keeps_original_identity():
    sentence = {"id": "long", "text": "a b c d e f g h", "section_id": "s", "source_element_ids": ["element"], "pages": [1]}
    chunks = pack_retrieval_chunks([sentence], token_counter=_counter(), embedding_token_limit=10)
    assert [chunk["sentence_ids"] for chunk in chunks] == [["long"], ["long"]]
    assert all(chunk["token_count"] <= 10 for chunk in chunks)
    assert [chunk["chunk_order"] for chunk in chunks] == [0, 1]


def test_long_unbroken_token_is_subdivided_to_the_token_budget():
    from app.services.embedding_tokenizer import _split_oversized_fragment

    count = lambda value: len(value)
    pieces = _split_oversized_fragment("x" * 23, 7, count)

    assert "".join(pieces) == "x" * 23
    assert all(count(piece) <= 7 for piece in pieces)


def test_hierarchy_keeps_flat_headings_when_no_depth_is_available():
    payload = {
        "elements": [
            {"element_id": "e1", "element_type": "section_header", "label": "section_header", "text": "First", "pages": [1], "raw": {}},
            {"element_id": "e2", "element_type": "text", "label": "text", "text": "Body", "pages": [1], "raw": {}},
            {"element_id": "e3", "element_type": "section_header", "label": "section_header", "text": "Second", "pages": [2], "raw": {}},
        ]
    }
    sections, elements = derive_hierarchy(payload)
    assert [section["title"] for section in sections] == ["First", "Second"]
    assert all(section["level"] == 1 for section in sections)
    assert elements[1]["section_id"] == sections[0]["section_id"]


def test_stable_canonical_projection_shape_is_json_compatible():
    class FakeDocument:
        def export_to_dict(self):
            return {"texts": [{"self_ref": "#/texts/0", "label": "text", "text": "Hello", "prov": []}], "body": {"children": [{"$ref": "#/texts/0"}]}}

    payload = build_canonical_payload(FakeDocument(), filename="input.pdf", source_metadata={"original_url": "https://example.test"})
    assert payload["schema_version"] == CANONICAL_SCHEMA_VERSION
    assert payload["elements"][0]["element_id"]
    assert payload["source_metadata"]["original_url"].startswith("https://")


def test_nested_word_children_collapse_to_one_block_element():
    class FakeDocument:
        def export_to_dict(self):
            return {
                "groups": [{
                    "self_ref": "#/groups/0",
                    "label": "list_item",
                    "children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}, {"$ref": "#/texts/2"}],
                }],
                "texts": [
                    {"self_ref": "#/texts/0", "label": "text", "text": "Senior", "parent": {"$ref": "#/groups/0"}, "prov": []},
                    {"self_ref": "#/texts/1", "label": "text", "text": "Software", "parent": {"$ref": "#/groups/0"}, "prov": []},
                    {"self_ref": "#/texts/2", "label": "text", "text": "Engineer", "parent": {"$ref": "#/groups/0"}, "prov": []},
                ],
                "body": {"children": [{"$ref": "#/groups/0"}]},
            }

    payload = build_canonical_payload(FakeDocument(), filename="resume.pdf", document_identity="resume")
    assert len(payload["elements"]) == 1
    assert payload["elements"][0]["text"] == "Senior Software Engineer"
    assert payload["elements"][0]["element_type"] == "list_item"


def test_text_with_word_children_keeps_parent_block_and_drops_children():
    class FakeDocument:
        def export_to_dict(self):
            return {
                "texts": [
                    {
                        "self_ref": "#/texts/0",
                        "label": "text",
                        "text": "Masters 2015 The University of Texas",
                        "children": [{"$ref": "#/texts/1"}, {"$ref": "#/texts/2"}],
                        "prov": [],
                    },
                    {"self_ref": "#/texts/1", "label": "text", "text": "Masters", "parent": {"$ref": "#/texts/0"}, "prov": []},
                    {"self_ref": "#/texts/2", "label": "text", "text": "2015", "parent": {"$ref": "#/texts/0"}, "prov": []},
                ],
                "body": {"children": [{"$ref": "#/texts/0"}]},
            }

    payload = build_canonical_payload(FakeDocument(), filename="resume.pdf")
    assert [item["text"] for item in payload["elements"]] == ["Masters 2015 The University of Texas"]


def _native_table_document():
    from docling_core.types.doc import DoclingDocument, TableCell, TableData

    def cell(text, row, col, *, row_span=1, col_span=1, column_header=False, row_header=False):
        return TableCell(
            text=text,
            start_row_offset_idx=row,
            end_row_offset_idx=row + row_span,
            start_col_offset_idx=col,
            end_col_offset_idx=col + col_span,
            column_header=column_header,
            row_header=row_header,
        )

    document = DoclingDocument(name="native-table")
    document.add_table(data=TableData(
        num_rows=4,
        num_cols=3,
        table_cells=[
            cell("Metric", 0, 0, column_header=True),
            cell("2024", 0, 1, column_header=True),
            cell("2025", 0, 2, column_header=True),
            cell("Revenue", 1, 0, row_header=True),
            cell("10", 1, 1),
            cell("12", 1, 2),
            cell("Total", 2, 0, col_span=3, row_header=True),
            cell("Footer", 3, 0, col_span=3),
        ],
    ))
    return document


def test_table_projection_uses_native_docling_headers_and_positions():
    element = build_canonical_payload(
        _native_table_document(), filename="table.pdf", document_identity="file"
    )["elements"][0]
    structure = element["table_structure"]
    assert structure["headers"] == ["Metric", "2024", "2025"]
    assert structure["header_rows"] == [0]
    assert structure["rows"][1] == ["Revenue", "10", "12"]
    assert structure["cells"][6]["col_span"] == 3
    assert "Metric: Revenue | 2024: 10 | 2025: 12" in element["text"]


def test_table_projection_does_not_invent_headers_for_native_cells():
    from docling_core.types.doc import DoclingDocument, TableCell, TableData

    document = DoclingDocument(name="no-header-table")
    document.add_table(data=TableData(
        num_rows=1,
        num_cols=2,
        table_cells=[
            TableCell(text="A", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=0, end_col_offset_idx=1),
            TableCell(text="B", start_row_offset_idx=0, end_row_offset_idx=1, start_col_offset_idx=1, end_col_offset_idx=2),
        ],
    ))
    structure = build_canonical_payload(document, filename="table.pdf")["elements"][0]["table_structure"]
    assert structure["headers"] == ["", ""]
    assert structure["header_rows"] == []
    assert structure["rows"] == [["A", "B"]]


def test_pipeline_versions_are_extraction_v5_and_table_pack_v6():
    from app.services.document_pipeline import EXTRACTION_PIPELINE_VERSION, RETRIEVAL_CHUNKING_VERSION

    assert EXTRACTION_PIPELINE_VERSION == "docling-pdf-v5"
    assert RETRIEVAL_CHUNKING_VERSION == "table-pack-v6"


def test_oversized_table_rows_prefix_headers_once_and_preserve_synthetic_provenance():
    payload = {
        "elements": [{
            "element_id": "table-1",
            "element_type": "table",
            "label": "table",
            "text": "Table headers: Metric | Value",
            "pages": [4],
            "table_structure": {
                "headers": ["Metric", "Value"],
                "rows": [["Revenue", "one two three four five six seven eight nine ten eleven twelve"]],
                "cells": [],
            },
        }]
    }

    sentences = project_sentences(payload, sentence_splitter=lambda value: [value], element_policy=None)
    assert sentences[0]["table_row_id"] == "table-1:row:1"
    assert sentences[0]["text"].startswith("Row 1:")
    assert "Table headers:" not in sentences[0]["text"]
    assert sentences[0]["source_spans"] == []
    assert sentences[0]["alignment_precision"] == "synthetic"

    chunks = pack_retrieval_chunks(
        sentences,
        token_counter=_counter(),
        embedding_token_limit=12,
        document_identity="file:generation",
    )
    assert len(chunks) > 1
    assert all(chunk["body_text"].count("Table headers: Metric | Value") == 1 for chunk in chunks)
    assert all(chunk["body_text"].count("Row 1:") == 1 for chunk in chunks)
    assert all(chunk["table_row_id"] == "table-1:row:1" for chunk in chunks)
    assert all(chunk["source_spans"] == [] for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


def test_table_chunks_share_one_header_and_flush_before_the_table_pack_ceiling():
    rows = [{"id": index, "text": f"Row {index}: value-{index}", "section_id": "s", "table_id": "table-1", "table_headers": "Method | Score", "table_row_prefix": f"Row {index}: ", "table_row_body": f"value-{index}"} for index in range(1, 80)]
    chunks = pack_retrieval_chunks(rows, token_counter=_counter(), embedding_token_limit=512)
    assert len(chunks) > 1
    assert all(chunk["body_text"].startswith("Table headers: Method | Score\nRow ") for chunk in chunks)
    assert all(chunk["body_text"].count("Table headers:") == 1 for chunk in chunks)
    assert all(chunk["token_count"] <= 192 for chunk in chunks)


def test_sibling_word_leaves_merge_into_one_block():
    class FakeDocument:
        def export_to_dict(self):
            return {
                "texts": [
                    {"self_ref": "#/texts/0", "label": "text", "text": "The", "prov": [{"page_no": 1}]},
                    {"self_ref": "#/texts/1", "label": "text", "text": "University", "prov": [{"page_no": 1}]},
                    {"self_ref": "#/texts/2", "label": "text", "text": "This is a complete sentence.", "prov": [{"page_no": 1}]},
                ],
                "body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/texts/1"}, {"$ref": "#/texts/2"}]},
            }

    payload = build_canonical_payload(FakeDocument(), filename="paper.pdf")
    assert [item["text"] for item in payload["elements"]] == [
        "The University",
        "This is a complete sentence.",
    ]


def test_consecutive_duplicate_tables_are_collapsed():
    from docling_core.types.doc import DoclingDocument, TableCell, TableData

    def table_data():
        return TableData(
            num_rows=1,
            num_cols=1,
            table_cells=[
                TableCell(
                    text="Only",
                    start_row_offset_idx=0,
                    end_row_offset_idx=1,
                    start_col_offset_idx=0,
                    end_col_offset_idx=1,
                ),
            ],
        )

    document = DoclingDocument(name="dup-tables")
    document.add_table(data=table_data())
    document.add_table(data=table_data())
    payload = build_canonical_payload(document, filename="tables.pdf")
    assert len([item for item in payload["elements"] if item["label"] == "table"]) == 1


def test_document_identity_prevents_cross_file_element_and_chunk_collisions():
    class FakeDocument:
        def export_to_dict(self):
            return {"texts": [{"self_ref": "#/texts/0", "label": "text", "text": "Same", "prov": []}], "body": {"children": [{"$ref": "#/texts/0"}]}}

    first = build_canonical_payload(FakeDocument(), filename="input.pdf", document_identity="file-a")
    second = build_canonical_payload(FakeDocument(), filename="input.pdf", document_identity="file-b")
    assert first["elements"][0]["element_id"] != second["elements"][0]["element_id"]
    sentence = {"id": 0, "text": "Same", "section_id": None, "source_element_ids": ["e"]}
    first_chunk = pack_retrieval_chunks([sentence], token_counter=_counter(), document_identity="file-a")
    second_chunk = pack_retrieval_chunks([sentence], token_counter=_counter(), document_identity="file-b")
    assert first_chunk[0]["chunk_id"] != second_chunk[0]["chunk_id"]
    model_a_chunk = pack_retrieval_chunks([sentence], token_counter=_counter(), document_identity="file-a:generation")
    model_b_chunk = pack_retrieval_chunks([sentence], token_counter=_counter(), document_identity="file-a:generation")
    assert model_a_chunk[0]["chunk_id"] == model_b_chunk[0]["chunk_id"]


def test_contextual_prefix_contains_document_title_and_is_bounded():
    sentence = {"id": "s", "text": "evidence", "section_id": "section", "heading_path": ["Results"]}
    first = pack_retrieval_chunks(
        [sentence], token_counter=_counter(), embedding_token_limit=20,
        document_identity="file-a", document_title="Annual Report A",
    )[0]
    second = pack_retrieval_chunks(
        [sentence], token_counter=_counter(), embedding_token_limit=20,
        document_identity="file-b", document_title="Annual Report B",
    )[0]
    assert "Document: Annual Report A" in first["contextualized_text"]
    assert "Document: Annual Report B" in second["contextualized_text"]
    assert first["contextualized_text"] != second["contextualized_text"]
    assert first["token_count"] <= 20


def test_document_title_is_part_of_chunking_identity():
    from types import SimpleNamespace
    from app.services.document_projection_service import retrieval_chunking_fingerprint

    base = SimpleNamespace(
        generation="generation-1",
        document_json={"filename": "canonical.pdf"},
        source_metadata_json={"original_title": "Report A"},
    )
    other = SimpleNamespace(
        generation="generation-1",
        document_json={"filename": "canonical.pdf"},
        source_metadata_json={"original_title": "Report B"},
    )
    assert retrieval_chunking_fingerprint(base, "model", "tokenizer") != retrieval_chunking_fingerprint(other, "model", "tokenizer")


def test_overflow_keeps_each_sentence_provenance_and_precise_fragment_spans():
    sentence = {
        "id": "long",
        "text": "one two three four five six seven eight",
        "section_id": "s",
        "paragraph_id": "p1",
        "source_element_ids": ["e1", "e2"],
        "pages": [1, 2],
        "source_spans": [
            {"element_id": "e1", "start": 0, "end": 13},
            {"element_id": "e2", "start": 14, "end": 39},
        ],
    }
    chunks = pack_retrieval_chunks([sentence], token_counter=_counter(), embedding_token_limit=8)
    assert all(chunk["sentence_ids"] == ["long"] for chunk in chunks)
    assert chunks[0]["source_element_ids"] == ["e1", "e2"]
    assert chunks[1]["source_element_ids"] == ["e2"]
    assert chunks[0]["source_spans"][0]["start"] == 0
    assert chunks[1]["source_spans"][0]["start"] > chunks[0]["source_spans"][0]["end"]


def test_paragraph_boundaries_and_chunk_tags_are_preserved():
    sentences = [
        {"id": 1, "text": "prose", "section_id": "s", "paragraph_id": "p1", "element_type": "text", "tags": ["text"], "source_element_ids": ["e1"]},
        {"id": 2, "text": "table row", "section_id": "s", "paragraph_id": "table", "table_id": "table", "element_type": "table", "tags": ["table"], "source_element_ids": ["e2"]},
    ]
    chunks = pack_retrieval_chunks(sentences, token_counter=_counter(), embedding_token_limit=32)
    assert [chunk["sentence_ids"] for chunk in chunks] == [["1"], ["2"]]
    assert chunks[0]["tags"] == ["text"]
    assert chunks[1]["tags"] == ["table"]


def test_adjacent_section_text_is_packed_until_the_embedding_budget():
    from app.services.document_pipeline import project_sentences

    payload = {"elements": [
        {"element_id": "e1", "element_type": "text", "text": "First.", "parent_element_id": "group"},
        {"element_id": "e2", "element_type": "text", "text": "Second.", "parent_element_id": "group"},
    ]}
    sentences = project_sentences(payload, sentence_splitter=lambda value: [value], element_policy=None)
    chunks = pack_retrieval_chunks(sentences, token_counter=_counter(), embedding_token_limit=32)
    assert [chunk["sentence_ids"] for chunk in chunks] == [["0", "1"]]
    assert chunks[0]["body_text"] == "First. Second."


def test_source_ids_are_schema_safe_and_model_independent():
    source_id = stable_source_id("file", "generation", "chunk")
    assert source_id.startswith("src_")
    assert "/" not in source_id
    assert "BAAI" not in source_id


def test_reading_projection_excludes_furniture_and_repeated_sentences_keep_offsets():
    from app.services.document_pipeline import project_sentences

    payload = {
        "elements": [
            {"element_id": "header", "element_type": "page_header", "label": "page_header", "text": "Header", "raw": {"content_layer": "furniture"}},
            {"element_id": "body", "element_type": "text", "label": "text", "text": "Repeat. Repeat.", "raw": {}},
            {"element_id": "caption", "element_type": "caption", "label": "caption", "text": "Caption", "raw": {}},
        ]
    }

    reading = project_sentences(payload, sentence_splitter=lambda value: ["Repeat.", "Repeat."] if value.startswith("Repeat") else [value])
    assert [item["text"] for item in reading] == ["Repeat.", "Repeat."]
    assert [item["source_spans"][0]["start"] for item in reading] == [0, 8]

    retrieval = project_sentences(payload, sentence_splitter=lambda value: [value], element_policy=None)
    assert [item["element_type"] for item in retrieval] == ["page_header", "text", "caption"]


def test_oversized_nonfirst_sentence_translates_fragment_offsets_to_element_coordinates():
    sentence = {
        "id": "second",
        "text": "one two three four five six",
        "section_id": "s",
        "paragraph_id": "p2",
        "source_element_ids": ["element"],
        "source_spans": [{"element_id": "element", "start": 100, "end": 128}],
    }
    chunks = pack_retrieval_chunks([{"id": "first", "text": "intro", "section_id": "s", "paragraph_id": "p1"}, sentence], token_counter=_counter(), embedding_token_limit=7)
    second_chunks = [chunk for chunk in chunks if chunk["sentence_ids"] == ["second"]]
    assert len(second_chunks) == 2
    assert second_chunks[0]["source_spans"][0]["start"] == 100
    assert second_chunks[1]["source_spans"][0]["start"] > second_chunks[0]["source_spans"][0]["end"]


def test_embedding_document_and_query_formatting_are_explicit():
    config = EmbeddingTokenizerConfig(
        identity="fixture-tokenizer",
        revision="r1",
        effective_input_limit=32,
        required_prefix="document: ",
        required_suffix=" </document>",
        query_prefix="query: ",
        query_suffix=" </query>",
    )
    assert config.format_document_input("body") == "document: body </document>"
    assert config.format_query_input("body") == "query: body </query>"
    assert config.fingerprint.endswith(":query: : </query>")


def test_tokenizer_failure_never_uses_whitespace_fallback(monkeypatch):
    import sys
    from types import SimpleNamespace
    from app.services.embedding_tokenizer import EmbeddingTokenizerUnavailableError, resolve_embedding_tokenizer

    class BrokenAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            raise OSError("fixture tokenizer unavailable")

    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "model-a")
    monkeypatch.setenv("EMBEDDING_TOKENIZER_CONFIG_JSON", "")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=BrokenAutoTokenizer))
    resolve_embedding_tokenizer.cache_clear()
    with pytest.raises(EmbeddingTokenizerUnavailableError):
        resolve_embedding_tokenizer("external-embed")


def test_local_embedding_tokenizer_uses_loaded_sentence_transformer(monkeypatch):
    from types import SimpleNamespace
    from app.services.embedding_tokenizer import resolve_embedding_tokenizer

    class FakeTokenizer:
        def __call__(self, formatted, add_special_tokens=True, truncation=False):
            return {"input_ids": list(formatted)}

    monkeypatch.setenv("LOCAL_EMBEDDING_MODEL", "model-a")
    monkeypatch.setenv("LOCAL_EMBEDDING_TOKENIZER", "model-a")
    monkeypatch.setenv("LOCAL_EMBEDDING_INPUT_LIMIT", "32")
    monkeypatch.setattr(
        "app.models.llm_server_client.get_local_embedding_model",
        lambda _name: SimpleNamespace(model=SimpleNamespace(tokenizer=FakeTokenizer())),
    )
    resolve_embedding_tokenizer.cache_clear()
    config, counter = resolve_embedding_tokenizer("model-a")
    assert config.identity == "model-a"
    assert counter.count("ab") == 2


def test_sentence_model_failure_never_switches_to_another_splitter(monkeypatch):
    import spacy
    from app.services import document_pipeline
    from app.services.document_pipeline import SentencePipelineUnavailableError

    def broken_load(_name):
        raise OSError("fixture sentence model unavailable")

    monkeypatch.setattr(spacy, "load", broken_load)
    document_pipeline._sentence_nlp.cache_clear()
    with pytest.raises(SentencePipelineUnavailableError):
        document_pipeline.sentence_pipeline_identity()
