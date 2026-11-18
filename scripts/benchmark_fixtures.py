"""Resolve benchmark PDF paths: committed bundled CC BY copies or optional arXiv download."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
BUNDLED_DIR = REPO / "benchmarks" / "fixtures" / "bundled"
BUNDLED_MANIFEST = REPO / "benchmarks" / "fixtures" / "bundled_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bundled_manifest() -> dict[str, Any]:
    if not BUNDLED_MANIFEST.is_file():
        return {"papers": []}
    return json.loads(BUNDLED_MANIFEST.read_text(encoding="utf-8"))


def bundled_paper_for_file(fixture_file: str) -> dict[str, Any] | None:
    for paper in load_bundled_manifest().get("papers") or []:
        if fixture_file in (
            paper.get("fixture_file"),
            paper.get("id"),
            f"{paper.get('id')}.pdf",
        ):
            return paper
    return None


def fixture_path(fixture_file: str) -> Path:
    bundled = bundled_paper_for_file(fixture_file)
    if bundled:
        path = BUNDLED_DIR / str(bundled["fixture_file"])
        if not path.is_file():
            raise FileNotFoundError(f"missing bundled PDF: {path}")
        expected = str(bundled.get("sha256") or "").lower()
        if expected and _sha256(path) != expected:
            raise RuntimeError(f"{fixture_file}: bundled sha256 mismatch")
        return path
    from benchmark_papers import paper_path

    return paper_path(fixture_file.replace(".pdf", ""))


def indexing_timeout_seconds(fixture_file: str) -> float:
    bundled = bundled_paper_for_file(fixture_file)
    if bundled:
        pages = int(bundled.get("pages") or 20)
        return min(7200.0, max(600.0, pages * 25.0))
    from benchmark_papers import indexing_timeout_seconds as legacy_timeout

    return legacy_timeout(fixture_file)


def list_benchmark_fixtures(*, include_optional_arxiv: bool = False) -> list[str]:
    files = [str(p["fixture_file"]) for p in load_bundled_manifest().get("papers") or []]
    if include_optional_arxiv:
        from benchmark_papers import load_manifest

        for paper in load_manifest().get("papers") or []:
            name = str(paper.get("fixture_file") or "")
            if name and name not in files:
                files.append(name)
    return files


def ensure_optional_arxiv(fixture_file: str) -> Path:
    from benchmark_papers import ensure_paper, paper_for_fixture

    paper = paper_for_fixture(fixture_file)
    if not paper:
        raise FileNotFoundError(fixture_file)
    return ensure_paper(paper)