#!/usr/bin/env python3
"""Capture localhost:3000 chat UI for each row in live_auto_proof.json."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROOF = Path("/opt/cursor/artifacts/live_auto_proof.json")
OUT_DIR = Path("/opt/cursor/artifacts")
BASE = "http://127.0.0.1:3000"


def main() -> int:
    if not PROOF.is_file():
        print(f"missing {PROOF}", file=sys.stderr)
        return 1
    rows = json.loads(PROOF.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()
        page.goto(BASE, wait_until="networkidle", timeout=120_000)
        time.sleep(2)
        for row in rows:
            case = str(row.get("case") or "").strip()
            if not case or case == "f":
                continue
            label = str(row.get("thread_name") or "").strip()
            if not label:
                continue
            needle = (
                "e2e-simple-2pdf-"
                if case == "simple-2pdf"
                else f"e2e-case-{case}-"
            )
            out = OUT_DIR / f"ui_case_{case}.png"
            try:
                page.goto(BASE, wait_until="networkidle", timeout=120_000)
                time.sleep(2)
                thread_link = page.locator("li").filter(has_text=needle).first
                thread_link.scroll_into_view_if_needed(timeout=60_000)
                thread_link.click(timeout=30_000)
                time.sleep(4)
                page.screenshot(path=str(out), full_page=True)
                captured.append(str(out))
                row["screenshot"] = str(out)
                print(f"captured {case} -> {out}", file=sys.stderr)
            except Exception as exc:
                row["screenshot_error"] = str(exc)[:500]
                print(f"failed {case}: {exc}", file=sys.stderr)
        browser.close()
    PROOF.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"captured": captured}))
    return 0 if captured else 1


if __name__ == "__main__":
    raise SystemExit(main())