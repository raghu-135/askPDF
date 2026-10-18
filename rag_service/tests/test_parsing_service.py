#!/usr/bin/env python3
"""Manual smoke test for Docling parsing (used by scripts/run_tests.py)."""

import argparse
import json
import os
import sys

from app.services.parsing_service import parse_with_docling


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to a PDF file")
    args = parser.parse_args()
    pdf_path = args.pdf
    if not os.path.exists(pdf_path):
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        return 1
    with open(pdf_path, "rb") as handle:
        data = handle.read()
    filename = os.path.basename(pdf_path)
    parse_result = parse_with_docling(data, filename)
    if parse_result is None:
        print("ERROR: Docling parsing failed", file=sys.stderr)
        return 1
    cells = sum(len(page_cells) for page_cells in parse_result.word_cells_by_page.values())
    summary = {
        "filename": filename,
        "text_items": len(parse_result.document.texts),
        "pages": len(parse_result.page_sizes),
        "word_cells": cells,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
