"""Canvas compile must resolve document locators to sentence ids (no silent nulls)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agent.evidence_contract import canonical_source_id, format_document_source_locator
from app.models.canvas_draft import CanvasCompileError, CanvasPublishDraft
from app.services.canvas_compiler import compile_canvas_draft


def test_canonical_source_id_matches_runtime_citation_menu_format():
    file_hash = "a" * 32
    source = {
        "file_hash": file_hash,
        "chunk_id": "src_abc123def4567890abcdef1234567890",
        "sentence_ids": ["217", "218"],
        "page_start": 8,
    }
    assert canonical_source_id(source) == f"doc:{file_hash}:{format_document_source_locator(source)}"


@pytest.mark.asyncio
async def test_compile_raises_when_chunk_locator_does_not_resolve():
    file_hash = "b" * 32
    chunk_token = "src_stale_after_rechunk"
    source_id = f"doc:{file_hash}:c-{chunk_token}"
    draft = CanvasPublishDraft.model_validate({
        "layout": "matrix",
        "title": "Stale chunk",
        "claims": [{"text": "Claim", "source_ids": [source_id]}],
    })
    with (
        patch(
            "app.services.canvas_compiler._readable_sentence_ids_for_chunk",
            new=AsyncMock(return_value=(None, [], False)),
        ),
        pytest.raises(CanvasCompileError) as exc,
    ):
        await compile_canvas_draft(draft, thread_file_hashes={file_hash}, embedding_model="embed-model")
    assert exc.value.code == "unknown_chunk"


@pytest.mark.asyncio
async def test_citation_menu_source_id_compiles_to_canvas_citation_with_sentence_ids():
    file_hash = "c" * 32
    src_id = "src_abc123def4567890abcdef1234567890"
    source = {"file_hash": file_hash, "chunk_id": "chunk-table-2", "source_id": src_id}
    menu_source_id = canonical_source_id(source)
    assert menu_source_id == f"doc:{file_hash}:c-chunk-table-2"

    draft = CanvasPublishDraft.model_validate({
        "layout": "matrix",
        "title": "Table evidence",
        "claims": [{"text": "Transformer baseline", "source_ids": [menu_source_id]}],
    })
    with (
        patch(
            "app.services.canvas_compiler._readable_sentence_ids_for_chunk",
            new=AsyncMock(return_value=(217, [217, 218], True)),
        ),
        patch(
            "app.services.canvas_compiler._sentence_exists",
            new=AsyncMock(return_value=True),
        ),
    ):
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
    citation = sources.citations[0]
    assert citation.sentence_id == 217
    assert citation.sentence_ids == [217, 218]


@pytest.mark.asyncio
async def test_src_alias_resolves_through_embedding_chunk_lookup():
    file_hash = "d" * 32
    src_id = "src_abc123def4567890abcdef1234567890"
    chunk = SimpleNamespace(chunk_id="chunk-live", source_id=src_id, file_hash=file_hash)
    draft = CanvasPublishDraft.model_validate({
        "layout": "matrix",
        "title": "Src alias",
        "claims": [{"text": "Claim", "source_ids": [src_id]}],
    })
    with (
        patch("app.services.canvas_compiler.get_canonical_document_repo") as repo_factory,
        patch(
            "app.services.canvas_compiler._readable_sentence_ids_for_chunk",
            new=AsyncMock(return_value=(64, [64, 65], True)),
        ),
        patch(
            "app.services.canvas_compiler._sentence_exists",
            new=AsyncMock(return_value=True),
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
    assert sources.citations[0].sentence_ids == [64, 65]
