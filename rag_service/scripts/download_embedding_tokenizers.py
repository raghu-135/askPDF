#!/usr/bin/env python3
"""Prefetch Hugging Face tokenizers listed in the embedding model catalog."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    default_catalog = root / "config" / "embedding_models.json"
    if not os.environ.get("EMBEDDING_MODELS_CONFIG", "").strip() and default_catalog.is_file():
        os.environ.setdefault("EMBEDDING_MODELS_CONFIG", str(default_catalog))


def main() -> int:
    _bootstrap_path()
    from transformers import AutoTokenizer

    from app.services.embedding_model_catalog import unique_hf_tokenizer_identities

    identities = list(unique_hf_tokenizer_identities())
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