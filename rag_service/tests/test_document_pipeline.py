from app.services.document_pipeline import (
    TokenCounter,
    build_canonical_payload,
    derive_hierarchy,
    pack_retrieval_chunks,
    stable_source_id,
    whitespace_token_counter,
)


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
    chunks = pack_retrieval_chunks(sentences, token_counter=_counter(), embedding_token_limit=8)
    assert chunks[0]["sentence_ids"] == ["1", "2", "3"]
    assert chunks[-1]["section_id"] == "s2"
    assert all(chunk["token_count"] <= 8 for chunk in chunks)
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
    chunks = pack_retrieval_chunks(sentences, token_counter=_counter())
    assert [chunk["sentence_ids"] for chunk in chunks] == [["1", "2"], ["3", "4"]]


def test_long_sentence_splits_deterministically_but_keeps_original_identity():
    sentence = {"id": "long", "text": "a b c d e f g h", "section_id": "s", "source_element_ids": ["element"], "pages": [1]}
    chunks = pack_retrieval_chunks([sentence], token_counter=_counter(), embedding_token_limit=4)
    assert [chunk["sentence_ids"] for chunk in chunks] == [["long"], ["long"]]
    assert all(chunk["token_count"] <= 4 for chunk in chunks)
    assert [chunk["chunk_order"] for chunk in chunks] == [0, 1]


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
    assert payload["schema_version"] == "docling-canonical-v1"
    assert payload["elements"][0]["element_id"]
    assert payload["source_metadata"]["original_url"].startswith("https://")


def test_table_projection_is_row_and_header_aware():
    class FakeDocument:
        def export_to_dict(self):
            return {
                "tables": [{
                    "self_ref": "#/tables/0",
                    "label": "table",
                    "data": {"grid": [["Name", "Score"], ["Ada", "10"]]},
                    "prov": [],
                }],
                "body": {"children": [{"$ref": "#/tables/0"}]},
            }

    payload = build_canonical_payload(FakeDocument(), filename="table.pdf", document_identity="file")
    element = payload["elements"][0]
    assert element["text"] == "Table headers: Name | Score\nRow 1: Ada | 10"
    assert element["table_structure"] == {"headers": ["Name", "Score"], "rows": [["Ada", "10"]]}


def test_document_identity_prevents_cross_file_element_and_chunk_collisions():
    class FakeDocument:
        def export_to_dict(self):
            return {"texts": [{"self_ref": "#/texts/0", "label": "text", "text": "Same", "prov": []}], "body": {"children": [{"$ref": "#/texts/0"}]}}

    first = build_canonical_payload(FakeDocument(), filename="input.pdf", document_identity="file-a")
    second = build_canonical_payload(FakeDocument(), filename="input.pdf", document_identity="file-b")
    assert first["elements"][0]["element_id"] != second["elements"][0]["element_id"]
    sentence = {"id": 0, "text": "Same", "section_id": None, "source_element_ids": ["e"]}
    first_chunk = pack_retrieval_chunks([sentence], document_identity="file-a")
    second_chunk = pack_retrieval_chunks([sentence], document_identity="file-b")
    assert first_chunk[0]["chunk_id"] != second_chunk[0]["chunk_id"]
    model_a_chunk = pack_retrieval_chunks([sentence], document_identity="file-a:generation")
    model_b_chunk = pack_retrieval_chunks([sentence], document_identity="file-a:generation")
    assert model_a_chunk[0]["chunk_id"] == model_b_chunk[0]["chunk_id"]


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
    chunks = pack_retrieval_chunks([sentence], token_counter=whitespace_token_counter(), embedding_token_limit=4)
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
    chunks = pack_retrieval_chunks(sentences)
    assert [chunk["sentence_ids"] for chunk in chunks] == [["1"], ["2"]]
    assert chunks[0]["tags"] == ["text"]
    assert chunks[1]["tags"] == ["table"]


def test_source_ids_are_schema_safe_and_model_independent():
    source_id = stable_source_id("file", "generation", "chunk")
    assert source_id.startswith("src_")
    assert "/" not in source_id
    assert "BAAI" not in source_id
