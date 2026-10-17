"""Canvas compile must resolve document locators to sentence ids (no silent nulls)."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.agent.evidence_contract import canonical_source_id, format_document_source_locator
from app.models.canvas_draft import CanvasCompileError, CanvasPublishDraft
from app.services.canvas_compiler import (
    _catalog_ids_from_chunk_rows,
    _readable_sentence_ids_for_chunk,
    _resolve_claim_sentence_ids,
    _sentence_claim_overlap_score,
    _sentence_ids_from_chunk_row,
    compile_canvas_draft,
)


def test_chunk_sentence_ids_prefer_claim_overlap_over_first_readable():
    row = SimpleNamespace(sentence_ids=[217, 218, 226, 227, 228])
    by_id = {
        217: {"text": "Table 2: BLEU on WMT English-German", "readable": True},
        218: {"text": "ByteNet 23.75", "readable": True},
        226: {"text": "Transformer (base) 27.3", "readable": True},
        227: {"text": "Transformer (big) 28.4", "readable": True},
        228: {"text": "Transformer (big) 41.0 on EN-FR", "readable": True},
    }
    claim = "Transformer big model 28.4 En-De"
    scroll_id, highlight_ids = _sentence_ids_from_chunk_row(row, by_id, claim_text=claim)
    assert scroll_id == 227
    assert highlight_ids == [227]
    assert 217 not in highlight_ids
    assert 218 not in highlight_ids
    assert _sentence_claim_overlap_score(claim, by_id[227]["text"]) > _sentence_claim_overlap_score(
        claim, by_id[218]["text"],
    )


@pytest.mark.asyncio
async def test_table_chunk_claim_matching_searches_sibling_table_rows():
    file_hash = "e" * 32
    header_row = SimpleNamespace(
        sentence_ids=[217, 218],
        table_id="table-2",
        chunk_order=24,
    )
    transformer_row = SimpleNamespace(
        sentence_ids=[226, 227, 228],
        table_id="table-2",
        chunk_order=32,
    )
    by_id = {
        217: {"text": "Table 2 header", "readable": True},
        218: {"text": "ByteNet 23.75", "readable": True},
        226: {"text": "Transformer (base) 27.3", "readable": True},
        227: {"text": "Transformer (big) 28.4", "readable": True},
        228: {"text": "Transformer (big) 41.0", "readable": True},
    }
    with (
        patch(
            "app.services.canvas_compiler._chunk_row_for_locator",
            new=AsyncMock(return_value=header_row),
        ),
        patch(
            "app.services.canvas_compiler._chunk_rows_for_table_id",
            new=AsyncMock(return_value=[header_row, transformer_row]),
        ),
        patch(
            "app.services.canvas_compiler._parsed_sentence_catalog",
            new=AsyncMock(return_value=by_id),
        ),
    ):
        scroll_id, highlight_ids, found = await _readable_sentence_ids_for_chunk(
            file_hash,
            "c-chunk-24",
            claim_text="Transformer big 28.4 En-De",
        )
    assert found is True
    assert scroll_id == 227
    assert 227 in highlight_ids
    assert 218 not in highlight_ids
    assert len(highlight_ids) == 1


def test_table_claim_trim_keeps_one_row_per_claim_number():
    rows = [
        SimpleNamespace(sentence_ids=[str(i) for i in range(217, 229)]),
    ]
    by_id = {
        217: {"text": "Table 2 BLEU", "readable": True},
        218: {"text": "ByteNet 23.75", "readable": True},
        226: {"text": "Transformer base 27.3", "readable": True},
        227: {"text": "Transformer big 28.4", "readable": True},
        228: {"text": "Training took 3.5 days on 8 P100 GPUs", "readable": True},
    }
    catalog_ids, readable_ids = _catalog_ids_from_chunk_rows(rows, by_id)
    _, bleu_ids = _resolve_claim_sentence_ids(
        catalog_ids,
        readable_ids,
        by_id,
        "Transformer big 28.4 BLEU",
    )
    assert bleu_ids == [227]
    _, cost_ids = _resolve_claim_sentence_ids(
        catalog_ids,
        readable_ids,
        by_id,
        "Training took 3.5 days on 8 P100 GPUs",
    )
    assert cost_ids == [228]


def test_chunk_sentence_ids_fall_back_to_first_readable_without_claim_overlap():
    row = SimpleNamespace(sentence_ids=[10, 11])
    by_id = {
        10: {"text": "Header row", "readable": True},
        11: {"text": "Detail row", "readable": True},
    }
    scroll_id, highlight_ids = _sentence_ids_from_chunk_row(row, by_id, claim_text="")
    assert scroll_id == 10
    assert highlight_ids == [10, 11]


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
async def test_compile_repairs_stale_ordinal_chunk_locator():
    file_hash = "b" * 32
    stale_source_id = f"doc:{file_hash}:c-0"
    live_token = "chunk-page-1"
    draft = CanvasPublishDraft.model_validate({
        "layout": "matrix",
        "title": "Stale ordinal",
        "claims": [{"text": "Page one summary", "source_ids": [stale_source_id]}],
    })
    live_row = SimpleNamespace(chunk_id=live_token, source_id="src_live", chunk_order=0)

    async def fake_readable(file_hash_arg, locator, **kwargs):
        if locator == "0":
            return None, [], False
        if locator == live_token:
            return 12, [12, 13], True
        return None, [], False

    with (
        patch(
            "app.services.canvas_compiler._chunk_row_by_order_index",
            new=AsyncMock(return_value=live_row),
        ),
        patch(
            "app.services.canvas_compiler._readable_sentence_ids_for_chunk",
            new=AsyncMock(side_effect=fake_readable),
        ),
        patch(
            "app.services.canvas_compiler._sentence_exists",
            new=AsyncMock(return_value=True),
        ),
    ):
        spec = await compile_canvas_draft(draft, thread_file_hashes={file_hash}, embedding_model="embed-model")
    sources = next(
        block
        for section in spec.sections
        for block in section.blocks
        if block.type == "sources"
    )
    assert sources.citations[0].sentence_ids == [12, 13]


@pytest.mark.asyncio
async def test_compile_drops_unknown_source_id_when_sibling_resolves():
    file_hash = "b" * 32
    good_id = f"doc:{file_hash}:c-chunk-2"
    bad_id = f"doc:{file_hash}:c-invented"
    draft = CanvasPublishDraft.model_validate({
        "layout": "matrix",
        "title": "Mixed citations",
        "claims": [{"text": "Claim", "source_ids": [bad_id, good_id]}],
    })

    async def fake_resolve(source_id, **kwargs):
        if source_id == bad_id:
            raise CanvasCompileError("unknown_chunk", "missing")
        from app.models.canvas import CanvasCitation

        return (
            CanvasCitation(
                kind="document",
                label="Claim",
                file_hash=file_hash,
                sentence_id=2,
                sentence_ids=[2, 3],
            ),
            (file_hash, 2),
        )

    with patch("app.services.canvas_compiler._resolve_source_id", new=AsyncMock(side_effect=fake_resolve)):
        spec = await compile_canvas_draft(draft, thread_file_hashes={file_hash}, embedding_model="embed-model")
    sources = next(
        block
        for section in spec.sections
        for block in section.blocks
        if block.type == "sources"
    )
    assert len(sources.citations) == 1
    assert sources.citations[0].sentence_id == 2


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
