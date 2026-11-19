import json
import os
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_eval_support import (  # noqa: E402
    bundled_fixture_names,
    load_cases,
    resolve_profile,
)
from benchmark_fixtures import list_bundled_fixtures  # noqa: E402


def test_smoke_tier_has_about_five_cases():
    cases = load_cases(tier="smoke")
    assert 4 <= len(cases) <= 6
    assert all(c.get("tier") == "smoke" for c in cases)


def test_all_eval_fixtures_are_bundled():
    cases = load_cases(tier="all")
    holdout = json.loads((ROOT / "benchmarks/holdout_questions.json").read_text())["questions"]
    names = bundled_fixture_names([*cases, *holdout])
    assert set(names).issubset(set(list_bundled_fixtures()))


def test_free_profile_default_rpm():
    profile = resolve_profile("free")
    assert profile.parallel is False
    assert profile.min_interval_s == pytest.approx(3.0, rel=0.05)


def test_fast_profile_parallel():
    profile = resolve_profile("fast")
    assert profile.parallel is True
    assert profile.max_workers >= 1


def test_questions_have_no_optional_fixture_or_mem0():
    data = json.loads((ROOT / "benchmarks/questions.json").read_text())
    for case in data["questions"]:
        assert not case.get("optional_fixture"), case["id"]
        assert "mem0" not in case["id"].lower()


def test_wait_stack_ready_times_out_cleanly(monkeypatch):
    from benchmark_eval_support import wait_stack_ready  # noqa: E402

    class _Client:
        def get(self, path):  # noqa: ARG002
            req = httpx.Request("GET", "http://test")
            return httpx.Response(503, request=req, json={"status": "unavailable"})

        def close(self):
            return None

    monkeypatch.setattr("benchmark_eval_support.httpx.Client", lambda **_: _Client())
    with pytest.raises(TimeoutError, match="stack not ready"):
        wait_stack_ready(timeout_s=0.5)