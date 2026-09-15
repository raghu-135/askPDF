from app.models.canvas import parse_canvas_spec


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
                        "rows": [["Latency bound", "Yes", "Partial"]],
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
                        "edges": [{"source": "a", "target": "b"}],
                    },
                    {"type": "markdown", "text": "The latency bound is only stated in paper A."},
                ],
            }
        ],
    }


def test_parse_canvas_spec_accepts_supported_blocks():
    spec = parse_canvas_spec(valid_spec())
    assert spec.title == "Compare the two papers"
    assert spec.sections[0].blocks[0].type == "stat"


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
