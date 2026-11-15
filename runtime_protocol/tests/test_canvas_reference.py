from runtime_protocol.canvas_reference import (
    CANVAS_URL_QUERY_PARAM,
    extract_canvas_references,
    is_canvas_artifact_id,
    pick_quoted_canvas_supersedes_target,
)


def test_extract_canvas_references_reads_uuid_and_share_url():
    canvas_id = "41a7cdc2-1234-5678-9abc-def012345678"
    text = (
        f"Please revise canvas {canvas_id} and also see "
        f"http://127.0.0.1:3000/?{CANVAS_URL_QUERY_PARAM}={canvas_id}."
    )
    refs = extract_canvas_references(text)
    assert refs == [canvas_id.lower()]


def test_extract_canvas_references_ignores_file_hashes():
    file_hash = "3deaaae75be4cddfd0b34d7568ae5635"
    assert file_hash not in extract_canvas_references(file_hash)


def test_pick_quoted_canvas_supersedes_target_prefers_last_mention():
    first = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    second = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    refs = extract_canvas_references(f"touch {first} then {second}")
    picked, warnings = pick_quoted_canvas_supersedes_target(
        refs,
        current_canvas_ids={second},
    )
    assert picked == second
    assert not warnings


def test_pick_quoted_canvas_resolves_lineage_head():
    root = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    head = "dddddddd-dddd-dddd-dddd-dddddddddddd"
    picked, warnings = pick_quoted_canvas_supersedes_target(
        [root],
        current_canvas_ids={head},
        lineage_head_resolver=lambda _canvas_id: head,
    )
    assert picked == head
    assert "quoted_canvas_resolved_to_current_head" in warnings


def test_is_canvas_artifact_id():
    assert is_canvas_artifact_id("41a7cdc2-1234-5678-9abc-def012345678")
    assert not is_canvas_artifact_id("not-a-uuid")