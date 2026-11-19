#!/usr/bin/env python3
"""Generate evals/promptfoo/tests.generated.yaml from benchmarks/questions.json."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
QUESTIONS = REPO / "benchmarks" / "questions.json"
OUT = REPO / "evals" / "promptfoo" / "tests.generated.yaml"


def _asserts(case: dict) -> list[dict]:
    match = case.get("answer_match") or {}
    asserts: list[dict] = [
        {
            "type": "javascript",
            "value": "context.providerResponse?.metadata?.retrieval_hit === true",
        }
    ]
    if match.get("any_contains"):
        asserts.append({"type": "contains-any", "value": match["any_contains"]})
    if match.get("all_contains"):
        for token in match["all_contains"]:
            asserts.append({"type": "icontains", "value": token})
    if match.get("decline"):
        asserts.append({"type": "javascript", "value": "!/(attached documents do not|do not address)/i.test(output)"})
    assertions = case.get("assertions") if isinstance(case.get("assertions"), dict) else {}
    if assertions.get("no_internal_knowledge_note"):
        asserts.append({"type": "javascript", "value": "!/\\(internal knowledge/i.test(output)"})
    return asserts


def main() -> int:
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    tests = []
    for case in data.get("questions") or []:
        if case.get("turns"):
            continue
        if "baseline" not in (case.get("suites") or []):
            continue
        tests.append(
            {
                "description": case["id"],
                "vars": {
                    "question_id": case["id"],
                    "question": case["question"],
                    "expected_pages": case.get("expected_pages") or [],
                    "answer_match": case.get("answer_match") or {},
                    "case_json": json.dumps(case),
                },
                "assert": _asserts(case),
            }
        )
    OUT.write_text(yaml.safe_dump(tests, sort_keys=False), encoding="utf-8")
    print(f"wrote {OUT} ({len(tests)} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())