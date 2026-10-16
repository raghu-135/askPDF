from langgraph_runtime.agent.citation_display import (
    build_citation_display_map,
    doc_id_fallback_label,
    manifest_entry_display_label,
)


def test_manifest_entry_display_label_uses_file_name_and_page():
    label = manifest_entry_display_label({
        "source_id": "doc:abc123:3",
        "file_name": "zephyr.pdf",
        "page_number": 3,
    })
    assert label == "zephyr.pdf, page 3"


def test_build_citation_display_map_indexes_manifest_ids():
    mapping = build_citation_display_map([
        {"source_id": "doc:hash:0", "file_name": "zephyr.pdf", "page_number": 1},
    ])
    assert mapping["doc:hash:0"] == "zephyr.pdf, page 1"


def test_doc_id_fallback_label_parses_locator():
    assert doc_id_fallback_label("doc:abc:12") == "document, page 12"
