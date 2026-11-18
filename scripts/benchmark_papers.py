"""Download and verify benchmark PDFs (not committed — arXiv license)."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "benchmarks" / "fixtures" / "manifest.json"
DEFAULT_CACHE = REPO / ".benchmark-pdf-cache"


def cache_dir() -> Path:
    raw = os.getenv("ASKPDF_BENCHMARK_PDF_CACHE", "").strip()
    return Path(raw) if raw else DEFAULT_CACHE


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "askPDF-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        dest.write_bytes(response.read())


def ensure_paper(paper: dict[str, Any], *, force: bool = False) -> Path:
    arxiv_id = str(paper["arxiv_id"])
    dest = cache_dir() / f"{arxiv_id}.pdf"
    expected = str(paper["sha256"]).lower()
    pdf_url = str(paper["pdf_url"])
    if dest.is_file() and not force:
        if not expected or _sha256(dest) == expected:
            return dest
        raise RuntimeError(f"{arxiv_id}: cache sha256 mismatch (delete {dest} or use --force)")
    if dest.is_file() and force and _sha256(dest) == expected:
        return dest
    _download(pdf_url, dest)
    actual = _sha256(dest)
    if actual != expected:
        if not expected:
            return dest
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"{arxiv_id}: download sha256 {actual} != manifest {expected}")
    return dest


def ensure_all(*, include_holdout: bool = False, force: bool = False) -> dict[str, Path]:
    manifest = load_manifest()
    out: dict[str, Path] = {}
    for paper in manifest["papers"]:
        if paper.get("role") == "holdout" and not include_holdout:
            continue
        path = ensure_paper(paper, force=force)
        arxiv_id = str(paper["arxiv_id"])
        out[arxiv_id] = path
        out[str(paper.get("fixture_file") or f"{arxiv_id}.pdf")] = path
    return out


def paper_path(arxiv_or_file: str, *, include_holdout: bool = True) -> Path:
    """Resolve manifest entry by arxiv id or fixture filename."""
    manifest = load_manifest()
    for paper in manifest["papers"]:
        if arxiv_or_file in (paper["arxiv_id"], paper.get("fixture_file"), f"{paper['arxiv_id']}.pdf"):
            if paper.get("role") == "holdout" and not include_holdout:
                raise FileNotFoundError(f"holdout paper {arxiv_or_file} excluded")
            return ensure_paper(paper)
    raise FileNotFoundError(arxiv_or_file)


def fixture_path(fixture_file: str) -> Path:
    from benchmark_fixtures import fixture_path as resolve_fixture

    return resolve_fixture(fixture_file)


def paper_for_fixture(fixture_file: str) -> dict[str, Any] | None:
    manifest = load_manifest()
    for paper in manifest["papers"]:
        names = (
            paper["arxiv_id"],
            paper.get("fixture_file"),
            f"{paper['arxiv_id']}.pdf",
        )
        if fixture_file in names:
            return paper
    return None


def indexing_timeout_seconds(fixture_file: str) -> float:
    """Scale indexing wait for large papers (e.g. 144-page survey)."""
    paper = paper_for_fixture(fixture_file)
    if not paper:
        return 1200.0
    pages = int(paper.get("pages") or 30)
    return min(7200.0, max(1200.0, pages * 25.0))


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Download and verify benchmark PDFs from arXiv")
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download when checksum missing or wrong")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest()
    if args.list:
        for paper in manifest["papers"]:
            print(
                paper["arxiv_id"],
                paper.get("arxiv_version"),
                paper.get("pages"),
                paper.get("role"),
            )
        return 0
    mapping = ensure_all(include_holdout=args.include_holdout, force=args.force)
    print(json.dumps({k: str(v) for k, v in sorted(mapping.items())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())