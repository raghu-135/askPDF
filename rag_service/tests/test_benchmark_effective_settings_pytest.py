import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ASKPDF_REPO_DIR", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_effective_settings import supplement_from_thread_settings


def test_supplement_fills_missing_context_budget_from_thread():
    effective = {"context_window": 131072, "context_budget": None}
    thread = {"context_budget": {"document_ratio": 0.2, "recent_ratio": 0.22}}
    merged = supplement_from_thread_settings(thread, effective)
    assert merged["context_budget"]["document_ratio"] == 0.2