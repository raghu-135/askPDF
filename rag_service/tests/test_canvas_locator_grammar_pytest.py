from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.models.canvas_draft import CanvasPublishDraft
from app.services.canvas_compiler import _citation_for_document_locator, compile_canvas_draft


@pytest.mark.asyncio
async def test_chunk_and_sentence_locators_do_not_cross_resolve():
    file_hash = "a" * 32
    row = SimpleNamespace(sentence_ids=["0", "5"], chunk_id="chunk-1", source_id="chunk-1")
    parsed = {
        "sentences": [
            {"id": 0, "readable": False, "text": "Header"},
            {"id": 5, "readable": True, "text": "Body sentence."},
            {"id": 64, "readable": True, "text": "Other sentence."},
        ],
    }

    with (
        patch("app.services.canvas_compiler._chunk_row_for_locator", new=AsyncMock(return_value=row)),
        patch(
            "app.services.canvas_compiler.get_file_parsed_sentences",
            new=AsyncMock(return_value=parsed),
        ),
        patch("app.services.canvas_compiler._sentence_exists", new=AsyncMock(return_value=True)),
    ):
        chunk_citation, chunk_quote = await _citation_for_document_locator(
            file_hash,
            "c-chunk-1",
            default_label="Chunk",
        )
        sentence_citation, sentence_quote = await _citation_for_document_locator(
            file_hash,
            "s-64",
            default_label="Sentence",
        )

    assert chunk_quote == (file_hash, 5)
    assert chunk_citation.sentence_ids == [5]
    assert sentence_quote == (file_hash, 64)
    assert sentence_citation.sentence_id == 64
    assert sentence_citation.sentence_ids == [64]


@pytest.mark.asyncio
async def test_legacy_digit_locator_prefers_chunk_when_ambiguous():
    file_hash = "b" * 32
    row = SimpleNamespace(sentence_ids=["64"], chunk_id="64", source_id="src-64")
    parsed = {"sentences": [{"id": 64, "readable": True, "text": "Shared id."}]}

    async def sentence_exists(_file_hash: str, sentence_id: int) -> bool:
        return sentence_id == 64

    with (
        patch("app.services.canvas_compiler._chunk_row_for_locator", new=AsyncMock(return_value=row)),
        patch("app.services.canvas_compiler._chunk_locator_exists", new=AsyncMock(return_value=True)),
        patch(
            "app.services.canvas_compiler.get_file_parsed_sentences",
            new=AsyncMock(return_value=parsed),
        ),
        patch("app.services.canvas_compiler._sentence_exists", new=AsyncMock(side_effect=sentence_exists)),
    ):
        citation, quote = await _citation_for_document_locator(file_hash, "64", default_label="Legacy")

    assert quote == (file_hash, 64)
    assert citation.sentence_ids == [64]


@pytest.mark.asyncio
async def test_compile_resolves_prefixed_chunk_locator_to_body_sentence():
    file_hash = "c" * 32
    source_id = f"doc:{file_hash}:c-chunk-body"
    draft = CanvasPublishDraft.model_validate({
        "layout": "matrix",
        "title": "Chunk body",
        "claims": [{"text": "Claim", "source_ids": [source_id]}],
    })
    with (
        patch(
            "app.services.canvas_compiler._readable_sentence_ids_for_chunk",
            new=AsyncMock(return_value=(7, [7, 8])),
        ),
        patch(
            "app.services.canvas_compiler._sentence_exists",
            new=AsyncMock(return_value=True),
        ),
    ):
        spec = await compile_canvas_draft(draft, thread_file_hashes={file_hash})
    sources = next(
        block
        for section in spec.sections
        for block in section.blocks
        if block.type == "sources"
    )
    assert sources.citations[0].sentence_id == 7
    assert sources.citations[0].sentence_ids == [7, 8]
