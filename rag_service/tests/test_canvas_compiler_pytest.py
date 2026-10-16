from unittest.mock import AsyncMock, patch

import pytest

from app.models.canvas_draft import CanvasCompileError, CanvasPublishDraft
from app.services.canvas_compiler import compile_canvas_draft


@pytest.mark.asyncio
async def test_compile_matrix_draft_builds_sources_and_table():
    draft = CanvasPublishDraft.model_validate({
        "layout": "matrix",
        "title": "Evidence map",
        "claims": [
            {
                "text": "Claim A",
                "source_ids": ["web:https://example.com/paper"],
                "tone": "supported",
            },
        ],
    })
    spec = await compile_canvas_draft(draft, thread_file_hashes=set())
    assert spec.title == "Evidence map"
    assert any(block.type == "sources" for section in spec.sections for block in section.blocks)
    table_blocks = [
        block for section in spec.sections for block in section.blocks if block.type == "table"
    ]
    assert table_blocks


@pytest.mark.asyncio
async def test_compile_rejects_unknown_document_source_id():
    draft = CanvasPublishDraft.model_validate({
        "layout": "compare",
        "title": "Compare",
        "claims": [{"text": "X", "source_ids": [f"doc:{'b' * 32}:0"]}],
    })
    with pytest.raises(CanvasCompileError) as exc:
        await compile_canvas_draft(draft, thread_file_hashes={"a" * 32})
    assert exc.value.code == "unknown_document"


@pytest.mark.asyncio
async def test_compile_timeline_requires_time_on_claims():
    with pytest.raises(ValueError):
        CanvasPublishDraft.model_validate({
            "layout": "timeline",
            "title": "Timeline",
            "claims": [{"text": "Event", "source_ids": [f"web:https://example.com/path"]}],
        })


@pytest.mark.asyncio
async def test_compile_timeline_events():
    draft = CanvasPublishDraft.model_validate({
        "layout": "timeline",
        "title": "Chronology",
        "claims": [
            {
                "text": "Started",
                "time": "2020",
                "source_ids": ["web:https://example.com/a"],
            },
        ],
    })
    spec = await compile_canvas_draft(draft, thread_file_hashes=set())
    timeline = next(
        block
        for section in spec.sections
        for block in section.blocks
        if block.type == "timeline"
    )
    assert timeline.events[0].time == "2020"


@pytest.mark.asyncio
async def test_compile_resolves_chunk_locator_to_sentence_quote():
    file_hash = "a" * 32
    source_id = f"doc:{file_hash}:chunk-1"
    draft = CanvasPublishDraft.model_validate({
        "layout": "matrix",
        "title": "Chunk quote",
        "claims": [{"text": "From chunk", "source_ids": [source_id]}],
    })
    with (
        patch(
            "app.services.canvas_compiler._first_sentence_for_chunk",
            new=AsyncMock(return_value=2),
        ),
        patch(
            "app.services.canvas_compiler._sentence_exists",
            new=AsyncMock(return_value=True),
        ),
    ):
        spec = await compile_canvas_draft(draft, thread_file_hashes={file_hash})
    quote_blocks = [
        block for section in spec.sections for block in section.blocks if block.type == "quote"
    ]
    assert len(quote_blocks) == 1
    assert quote_blocks[0].sentence_id == 2


@pytest.mark.asyncio
async def test_compile_resolves_src_alias_to_document_citation():
    file_hash = "a" * 32
    src_id = "src_abc123def4567890abcdef1234567890"
    draft = CanvasPublishDraft.model_validate({
        "layout": "matrix",
        "title": "Retrieval ids",
        "claims": [{"text": "From search", "source_ids": [src_id]}],
    })
    chunk = type("Chunk", (), {"chunk_id": "3", "source_id": src_id, "file_hash": file_hash})()
    with (
        patch(
            "app.services.canvas_compiler.get_canonical_document_repo",
        ) as repo_factory,
        patch(
            "app.services.canvas_compiler._sentence_exists",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "app.services.canvas_compiler._first_sentence_for_chunk",
            new=AsyncMock(return_value=1),
        ),
    ):
        repo = AsyncMock()
        repo.get_chunks_by_source_id = AsyncMock(return_value=[chunk])
        repo_factory.return_value = repo
        spec = await compile_canvas_draft(
            draft,
            thread_file_hashes={file_hash},
            embedding_model="baai/bge-base-en-v1.5",
        )
    sources = next(
        block
        for section in spec.sections
        for block in section.blocks
        if block.type == "sources"
    )
    assert sources.citations[0].file_hash == file_hash


def test_publish_idempotency_key_differs_for_revision():
    from app.tools.publish_canvas import _publish_idempotency_key
    from app.models.canvas import CanvasSpec

    spec = CanvasSpec.model_validate({
        "schema_version": 1,
        "title": "T",
        "sections": [{"title": "S", "blocks": [{"type": "stat", "value": "1", "label": "Claims"}]}],
    })
    create_key = _publish_idempotency_key(
        requested=None,
        agent_run_id="run-1",
        supersedes_id=None,
        spec=spec,
    )
    revision_key = _publish_idempotency_key(
        requested=None,
        agent_run_id="run-1",
        supersedes_id="41a7cdc2-1234-5678-9abc-def012345678",
        spec=spec,
    )
    assert create_key == "publish_canvas:run-1:create"
    assert revision_key == "publish_canvas:run-1:41a7cdc2-1234-5678-9abc-def012345678"
    retry_key = _publish_idempotency_key(
        requested="publish_canvas:run-1:create",
        agent_run_id="run-1",
        supersedes_id=None,
        spec=spec,
    )
    assert retry_key == "publish_canvas:run-1:create"
