"""Docling parsing smoke tests (visualization helpers optional)."""

import os

import pytest

from app.services.parsing_service import parse_with_docling


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
    assert len(parse_result.document.texts) > 0


def test_docling_json_output(sample_pdf_data, sample_filename, tmp_path):
    parse_result = parse_with_docling(sample_pdf_data, sample_filename, debug_output_dir=str(tmp_path))
    assert parse_result is not None
    json_files = list(tmp_path.glob("docling_raw_output_*.json"))
    assert json_files
