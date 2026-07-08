from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


BUILTIN_WORKFLOW_DIR = Path(__file__).with_name("builtins")


@lru_cache(maxsize=1)
def _builtin_workflow_payloads() -> tuple[Dict[str, Any], ...]:
    payloads: list[Dict[str, Any]] = []
    for path in sorted(BUILTIN_WORKFLOW_DIR.glob("*.json")):
        builtin_key = path.stem
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("builtin_key") != builtin_key:
            raise ValueError(f"Builtin workflow file {path} has mismatched builtin_key")
        spec_json = payload.get("spec_json")
        if not isinstance(spec_json, dict) or spec_json.get("schema_version") != 2:
            raise ValueError(f"Builtin workflow file {path} must contain schema_version 2 spec_json")
        payloads.append(payload)
    return tuple(payloads)


def load_builtin_workflows() -> list[Dict[str, Any]]:
    return [deepcopy(payload) for payload in _builtin_workflow_payloads()]


def builtin_workflow_keys() -> frozenset[str]:
    return frozenset(str(payload["builtin_key"]) for payload in _builtin_workflow_payloads())
