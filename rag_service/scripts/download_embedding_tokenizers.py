#!/usr/bin/env python3
"""Prefetch Hugging Face tokenizers used by remote embedding models."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_registry():
    candidates = [
        Path("/tmp/embedding_tokenizer_registry.py"),
        Path(__file__).resolve().parents[1] / "app/services/embedding_tokenizer_registry.py",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("embedding_tokenizer_registry", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    raise FileNotFoundError("embedding_tokenizer_registry.py was not found")


def main() -> int:
    from transformers import AutoTokenizer

    registry = _load_registry()
    identities = list(registry.unique_tokenizer_identities())
    for identity, revision in identities:
        kwargs = {"local_files_only": False}
        if revision:
            kwargs["revision"] = revision
        print(f"Downloading tokenizer {identity}")
        AutoTokenizer.from_pretrained(identity, **kwargs)
    print(f"Downloaded {len(identities)} embedding tokenizers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
