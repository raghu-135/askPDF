"""Lightweight, parser-independent extraction contract metadata."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version as package_version


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.lower() == "true"


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _installed(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "unknown"


def extraction_configuration(*, sentence_model: str | None = None) -> dict[str, object]:
    """Return parser inputs without importing Docling, spaCy, or Torch."""
    return {
        "do_ocr": _env_bool("DOCLING_DO_OCR", False),
        "do_table_structure": _env_bool("DOCLING_DO_TABLE_STRUCTURE", True),
        "do_formula_enrichment": _env_bool("DOCLING_DO_FORMULA_ENRICHMENT", True),
        "table_mode": _env_str("DOCLING_TABLE_MODE", "FAST").upper(),
        "force_full_page_ocr": _env_bool("DOCLING_FORCE_FULL_PAGE_OCR", False),
        "docling": _installed("docling"),
        "docling_core": _installed("docling-core"),
        "pdfplumber": _installed("pdfplumber"),
        "spacy": _installed("spacy"),
        "sentence_model": sentence_model or "en_core_web_sm:configured",
    }


__all__ = ["extraction_configuration"]
