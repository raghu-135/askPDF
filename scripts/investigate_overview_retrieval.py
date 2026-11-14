#!/usr/bin/env python3
"""Single-model overview probe; multi-model comparison: compare_overview_models.py."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

DEFAULT_OUT = Path("/opt/cursor/artifacts/overview-investigation.json")


def main() -> int:
    model = os.getenv("ASKPDF_BENCHMARK_MODEL", "openai/gpt-4.1-mini")
    env = {**os.environ, "ASKPDF_OVERVIEW_MODELS": model, "ASKPDF_MODEL_COMPARE_OUT": str(DEFAULT_OUT)}
    script = Path(__file__).resolve().parent / "compare_overview_models.py"
    return subprocess.call([sys.executable, str(script)], env=env)


if __name__ == "__main__":
    raise SystemExit(main())