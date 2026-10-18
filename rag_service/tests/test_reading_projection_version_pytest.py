"""Reading projection overlay version and stale-cache republish."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.services import document_projection_service
from app.services.document_pipeline import CANONICAL_SCHEMA_VERSION, EXTRACTION_PIPELINE_VERSION, pack_retrieval_chunks
from app.services.file_processing_service import (
    READING_PROJECTION_VERSION,
    refresh_reading_projection_if_stale,
    republish_reading_projection_from_canonical,
)


@pytest.fixture(autouse=True)
def _sentence_pipeline_identity(monkeypatch):
    monkeypatch.setattr(
        document_projection_service,
        "sentence_pipeline_identity",
        lambda: "en_core_web_sm:test-double",
    )


def _token_counter():
    def count(value: str) -> int:
        return len(value.replace("\n", " ").split())

    def split(value: str, limit: int):
        words = value.split()
        return [" ".join(words[i : i + limit]) for i in range(0, len(words), limit)]

    from app.services.document_pipeline import TokenCounter

    return TokenCounter(count=count, split=split)


@pytest.mark.asyncio
async def test_stale_reading_version_republishes_with_new_boxes_without_conversion(monkeypatch):
    element_id = "tbl-1"
    canonical_sentences = [
        {
            "id": 0,
            "text": "Row 1: value",
            "element_type": "table",
            "table_row_id": f"{element_id}:row:1",
            "table_cell_ids": [f"{element_id}:cell:0"],
            "source_element_ids": [element_id],
            "bboxes": [],
        }
    ]
    document_json = {
        "schema_version": CANONICAL_SCHEMA_VERSION,
        "docling": {},
        "elements": [
            {
                "element_id": element_id,
                "element_type": "table",
                "label": "table",
                "pages": [1],
                "provenance": [
                    {
                        "page_no": 1,
                        "bbox": {"l": 72.0, "t": 700.0, "r": 500.0, "b": 600.0, "coord_origin": "BOTTOMLEFT"},
                    }
                ],
                "raw": {
                    "data": {
                        "table_cells": [
                            {
                                "start_row_offset_idx": 1,
                                "end_row_offset_idx": 2,
                                "start_col_offset_idx": 1,
                                "end_col_offset_idx": 2,
                                "row_header": False,
                                "text": "value",
                                "bbox": {"l": 100.0, "t": 120.0, "r": 200.0, "b": 130.0, "coord_origin": "TOPLEFT"},
                            }
                        ]
                    }
                },
                "table_structure": {"header_rows": [0], "rows": [["h"], ["value"]], "cells": []},
            }
        ],
        "sections": [],
        "reading_projection": canonical_sentences,
    }
    canonical = SimpleNamespace(
        file_hash="file-a",
        generation="generation-current",
        extraction_fingerprint="extract-current",
        status="completed",
        source_metadata_json={
            "_file_hash": "file-a",
            "_extraction_contract_fingerprint": document_projection_service._current_extraction_contract_fingerprint(),
            "_extraction_pipeline_version": EXTRACTION_PIPELINE_VERSION,
        },
        document_json=document_json,
    )
    stored_parsed = {
        "version": "2.0",
        "generation": "generation-current",
        "extraction_fingerprint": "extract-current",
        "sentences": [{"id": 0, "text": "Row 1: value", "bboxes": []}],
    }
    persisted: dict[str, str] = {}

    class Repo:
        async def get(self, file_hash):
            return canonical

    async def get_parsed(_hash):
        return stored_parsed

    async def update_parsed(file_hash, payload):
        persisted["payload"] = payload
        stored_parsed.update(json.loads(payload))
        return True

    conversion_calls = []
    embedding_calls = []

    monkeypatch.setattr(document_projection_service, "get_canonical_document_repo", lambda: Repo())
    monkeypatch.setattr("app.services.file_processing_service.get_canonical_document_repo", lambda: Repo())
    monkeypatch.setattr(document_projection_service, "get_file_parsed_sentences", get_parsed)
    monkeypatch.setattr("app.services.file_processing_service.get_file_parsed_sentences", get_parsed)
    monkeypatch.setattr("app.services.file_processing_service.update_file_parsed_sentences", update_parsed)
    monkeypatch.setattr(
        "app.services.file_processing_service.enqueue_pdf_conversion",
        AsyncMock(side_effect=lambda **_kwargs: conversion_calls.append(1)),
    )
    monkeypatch.setattr(
        "app.services.embedding_materialization_service.enqueue_document_embedding_if_needed",
        AsyncMock(side_effect=lambda **_kwargs: embedding_calls.append(1)),
    )

    freshness = await document_projection_service.evaluate_document_freshness("file-a", require_reading=True)
    assert freshness["canonical_ready"] is True
    assert freshness["reading_ready"] is False
    assert freshness["reason"] == "reading_stale"

    assert await refresh_reading_projection_if_stale("file-a") is True
    assert conversion_calls == []
    assert embedding_calls == []
    assert stored_parsed["version"] == READING_PROJECTION_VERSION
    assert stored_parsed["sentences"][0].get("bboxes")

    chunks_before = pack_retrieval_chunks(
        canonical_sentences,
        token_counter=_token_counter(),
        embedding_token_limit=128,
        document_identity="file-a:generation-current",
        document_title="doc",
    )
    chunks_after = pack_retrieval_chunks(
        stored_parsed["sentences"],
        token_counter=_token_counter(),
        embedding_token_limit=128,
        document_identity="file-a:generation-current",
        document_title="doc",
    )
    assert [chunk["chunk_id"] for chunk in chunks_before] == [chunk["chunk_id"] for chunk in chunks_after]


@pytest.mark.asyncio
async def test_republish_reading_projection_from_canonical_is_noop_when_current(monkeypatch):
    canonical = SimpleNamespace(
        file_hash="file-b",
        generation="g1",
        extraction_fingerprint="f1",
        status="completed",
        document_json={"reading_projection": [{"id": 0, "text": "x"}]},
    )
    current = {
        "version": READING_PROJECTION_VERSION,
        "generation": "g1",
        "extraction_fingerprint": "f1",
        "sentences": [{"id": 0, "text": "x", "bboxes": [{"page": 1, "x0": 1, "y0": 2, "x1": 3, "y1": 4}]}],
    }

    class Repo:
        async def get(self, _hash):
            return canonical

    update_calls = []

    async def update_parsed(_hash, _payload):
        update_calls.append(1)
        return True

    monkeypatch.setattr("app.services.file_processing_service.get_canonical_document_repo", lambda: Repo())
    monkeypatch.setattr("app.services.file_processing_service.get_file_parsed_sentences", AsyncMock(return_value=current))

    assert await republish_reading_projection_from_canonical("file-b") is True
    assert update_calls == []
