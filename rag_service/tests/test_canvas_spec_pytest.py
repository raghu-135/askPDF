from app.models.canvas import (
    CanvasAdmissionError,
    admit_canvas_spec,
    parse_canvas_spec,
    resolve_quote_blocks,
)


def valid_spec():
    return {
        "schema_version": 1,
        "title": "Compare the two papers",
        "summary": "Overlap and gaps across the attached sources.",
        "sections": [
            {
                "title": "Overlap",
                "blocks": [
                    {"type": "stat", "value": "3", "label": "Shared claims", "tone": "info"},
                    {
                        "type": "table",
                        "headers": ["Claim", "Paper A", "Paper B"],
                        "rows": [["Latency bound", "Yes", {"text": "Partial", "tone": "partial"}]],
                    },
                    {
                        "type": "callout",
                        "tone": "warning",
                        "title": "Unresolved",
                        "body": "The second paper never reports the same evaluation split.",
                    },
                    {
                        "type": "sources",
                        "citations": [
                            {"kind": "document", "label": "Paper A, sentence 2", "file_hash": "file-a", "sentence_id": 2},
                            {"kind": "web", "label": "Vendor note", "url": "https://example.com/note"},
                        ],
                    },
                    {
                        "type": "dag",
                        "title": "Evidence path",
                        "nodes": [{"id": "a", "label": "Claim"}, {"id": "b", "label": "Source"}],
                        "edges": [{"source": "a", "target": "b", "relation": "supports"}],
                    },
                    {"type": "markdown", "text": "The latency bound is only stated in paper A."},
                    {
                        "type": "quote",
                        "file_hash": "file-a",
                        "sentence_id": 2,
                        "label": "Definition",
                    },
                    {
                        "type": "timeline",
                        "events": [
                            {
                                "time": "2020",
                                "label": "First report",
                                "citation": {"kind": "web", "label": "Archive", "url": "https://example.com/2020"},
                            }
                        ],
                    },
                    {
                        "type": "claim_pair",
                        "relation": "contradicts",
                        "left": {
                            "text": "Paper A claims X.",
                            "citation": {"kind": "document", "label": "A", "file_hash": "file-a"},
                        },
                        "right": {
                            "text": "Paper B denies X.",
                            "citation": {"kind": "document", "label": "B", "file_hash": "file-b"},
                        },
                    },
                ],
            }
        ],
    }


def test_parse_canvas_spec_accepts_supported_blocks():
    spec = parse_canvas_spec(valid_spec())
    assert spec.title == "Compare the two papers"
    assert spec.sections[0].blocks[0].type == "stat"
    table = spec.sections[0].blocks[1]
    assert table.rows[0][2].tone == "partial"
    dag = next(block for block in spec.sections[0].blocks if block.type == "dag")
    assert dag.edges[0].relation == "supports"


def test_parse_canvas_spec_rejects_unknown_block_type():
    payload = valid_spec()
    payload["sections"][0]["blocks"].append({"type": "iframe", "src": "https://evil.example"})
    try:
        parse_canvas_spec(payload)
    except Exception as exc:
        assert "iframe" in str(exc) or "type" in str(exc)
    else:
        raise AssertionError("unknown block types must be rejected")


def test_parse_canvas_spec_rejects_javascript_url():
    payload = valid_spec()
    payload["sections"][0]["blocks"][3]["citations"][1]["url"] = "javascript:alert(1)"
    try:
        parse_canvas_spec(payload)
        raise AssertionError("javascript urls must be rejected")
    except Exception:
        pass


def test_parse_canvas_spec_rejects_ragged_table():
    payload = valid_spec()
    payload["sections"][0]["blocks"][1]["rows"] = [["only-one"]]
    try:
        parse_canvas_spec(payload)
        raise AssertionError("ragged tables must be rejected")
    except Exception:
        pass


def test_parse_canvas_spec_rejects_invalid_dag_relation():
    payload = valid_spec()
    dag = next(block for block in payload["sections"][0]["blocks"] if block["type"] == "dag")
    dag["edges"][0]["relation"] = "maybe"
    try:
        parse_canvas_spec(payload)
        raise AssertionError("invalid dag relation must be rejected")
    except Exception:
        pass


def test_admit_canvas_spec_rejects_unknown_document_on_quote():
    spec = parse_canvas_spec({
        "schema_version": 1,
        "title": "Quote only",
        "sections": [
            {
                "title": "Passage",
                "blocks": [{"type": "quote", "file_hash": "missing", "sentence_id": 0}],
            }
        ],
    })
    try:
        admit_canvas_spec(spec, thread_file_hashes={"doc1"})
        raise AssertionError("unknown quote file_hash must be rejected")
    except CanvasAdmissionError as exc:
        assert exc.code == "unknown_document_citation"


def test_admit_canvas_spec_allows_quote_without_sources_block():
    spec = parse_canvas_spec({
        "schema_version": 1,
        "title": "Quote only",
        "sections": [
            {
                "title": "Passage",
                "blocks": [{"type": "quote", "file_hash": "doc1", "sentence_id": 0}],
            }
        ],
    })
    admit_canvas_spec(spec, thread_file_hashes={"doc1"})


def test_resolve_quote_blocks_fills_text():
    spec = parse_canvas_spec({
        "schema_version": 1,
        "title": "Quote",
        "sections": [
            {
                "title": "Passage",
                "blocks": [{"type": "quote", "file_hash": "doc1", "sentence_id": 1}],
            }
        ],
    })
    resolved = resolve_quote_blocks(
        spec,
        sentences_by_file={"doc1": {"sentences": [{"id": 1, "text": "Hello world.", "page": 4}]}},
    )
    quote = resolved.sections[0].blocks[0]
    assert quote.text == "Hello world."
    assert quote.page == 4
