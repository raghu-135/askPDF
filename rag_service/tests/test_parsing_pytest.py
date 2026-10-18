#!/usr/bin/env python3
"""Pytest tests for rag-service parsing_service."""

import os

import pytest

from app.services.parsing_service import DoclingParseResult, extraction_configuration, parse_with_docling


@pytest.fixture
def sample_pdf_path():
    test_dir = os.path.dirname(__file__)
    return os.path.join(test_dir, "01030000000000.pdf")


@pytest.fixture
def sample_pdf_data(sample_pdf_path):
    with open(sample_pdf_path, "rb") as handle:
        return handle.read()


@pytest.fixture
def sample_filename(sample_pdf_path):
    return os.path.basename(sample_pdf_path)


def test_docling_parsing(sample_pdf_data, sample_filename):
    parse_result = parse_with_docling(sample_pdf_data, sample_filename)
    assert parse_result is not None
    assert isinstance(parse_result, DoclingParseResult)
    assert len(parse_result.document.texts) > 0
    assert parse_result.page_sizes
    total_cells = sum(len(cells) for cells in parse_result.word_cells_by_page.values())
    assert total_cells > 0


def test_extraction_configuration_exposes_cache_invalidating_options(monkeypatch):
    import app.services.parsing_service as parsing_service

    monkeypatch.setattr(parsing_service, "sentence_pipeline_identity", lambda: "en_core_web_sm:test")
    monkeypatch.setattr(parsing_service._pipeline_options, "do_ocr", False)
    without_ocr = extraction_configuration()
    monkeypatch.setattr(parsing_service._pipeline_options, "do_ocr", True)
    with_ocr = extraction_configuration()

    assert without_ocr["do_ocr"] is False
    assert with_ocr["do_ocr"] is True
    assert with_ocr["generate_parsed_pages"] is True
    assert {"do_table_structure", "do_formula_enrichment", "table_mode", "docling", "docling_core"} <= set(with_ocr)
