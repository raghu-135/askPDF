#!/usr/bin/env python3
"""Upload benchmark PDFs into a running stack."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_papers import ensure_all, fixture_path, load_manifest
from live_stack_client import create_thread, live_client, upload_and_wait

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "benchmarks" / "reports" / ".fixture_hashes.json"


def fixture_files(*, include_holdout: bool) -> list[str]:
    papers = load_manifest()["papers"]
    if include_holdout:
        return [str(p["fixture_file"]) for p in papers]
    return [str(p["fixture_file"]) for p in papers if p.get("role") != "holdout"]


def main() -> int:
    include_holdout = os.getenv("ASKPDF_INCLUDE_HOLDOUT", "0") == "1"
    thread_id = os.getenv("ASKPDF_BENCHMARK_THREAD_ID", "").strip()
    ensure_all(include_holdout=include_holdout)
    files = fixture_files(include_holdout=include_holdout)
    mapping: dict[str, str] = {}
    with live_client() as client:
        if not thread_id:
            thread_id = create_thread(client, f"benchmark-fixtures-{int(time.time())}")
        for name in files:
            file_hash = upload_and_wait(client, thread_id, fixture_path(name))
            mapping[name] = file_hash
            print(f"indexed {name} -> {file_hash}")
    payload = {"thread_id": thread_id, "files": mapping}
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())