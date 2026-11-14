#!/usr/bin/env python3
"""Capture live tool-approval UI on localhost:3000."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ART = Path("/opt/cursor/artifacts")
BASE = "http://127.0.0.1:3000"
MODEL = "openai/gpt-4.1-mini"


def set_web_ask(page) -> None:
    btn = page.locator(
        'button[aria-label*="Internet Search"], button[aria-label*="Ask me every time"]'
    ).first
    for _ in range(4):
        if "Ask me every time" in (btn.get_attribute("aria-label") or ""):
            return
        btn.click()
        time.sleep(0.4)


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    viewports = [("1440x900", 1440, 900), ("1024x768", 1024, 768)]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for suffix, w, h in viewports:
            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(BASE, wait_until="domcontentloaded", timeout=120_000)
            time.sleep(2)
            combo = page.get_by_role("combobox", name="Select LLM").first
            if combo.is_visible(timeout=5000):
                combo.click()
                combo.fill("gpt-4.1-mini")
                time.sleep(0.5)
                opt = page.get_by_role("option", name=MODEL).first
                if opt.count():
                    opt.click()
            page.get_by_placeholder("Ask a question about your documents").first.wait_for(timeout=180_000)
            set_web_ask(page)
            inp = page.get_by_placeholder("Ask a question about your documents").first
            inp.fill("Use the internet search tool to find the current USD to EUR exchange rate. Reply with one number.")
            inp.press("Enter")
            approve = page.get_by_role("button", name="Approve once")
            try:
                approve.wait_for(timeout=180_000)
            except Exception as exc:
                page.screenshot(path=str(ART / f"chat-tool-approval-{suffix}-live-missing-prompt.png"))
                print(f"{suffix}: approval never appeared: {exc}", file=sys.stderr)
                continue
            page.screenshot(path=str(ART / f"chat-tool-approval-{suffix}-prompt.png"))
            approve.click()
            cont = page.get_by_role("button", name="Approved, continuing")
            cont.wait_for(timeout=30_000)
            page.screenshot(path=str(ART / f"chat-tool-approval-{suffix}-continuing.png"))
            page.get_by_test_id("chat-human-review-panel").wait_for(state="hidden", timeout=240_000)
            page.get_by_text("EUR", exact=False).first.wait_for(timeout=240_000)
            time.sleep(1)
            page.screenshot(path=str(ART / f"chat-tool-approval-{suffix}-answer.png"))
            print(f"{suffix}: ok", file=sys.stderr)
            page.close()
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())