import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_scoring import INTERNAL_KNOWLEDGE_RE, grounding_pass, numeric_score, string_constraints


def test_numeric_range_not_single_digit_trap():
    answer = "The meeting was on 27–28 October 1927."
    assert numeric_score(answer, {"target": 27, "tolerance": 0}) is True


def test_internal_knowledge_fails_grounding():
    answer = "Foo [Source: x]. (Internal knowledge: not in PDF.)"
    assert grounding_pass(answer) is False
    assert INTERNAL_KNOWLEDGE_RE.search(answer)


def test_decline_match():
    answer = "The attached documents do not contain Neptune's orbital period."
    assert string_constraints(answer, {"decline": True}) is True