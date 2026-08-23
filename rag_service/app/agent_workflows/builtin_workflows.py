from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from app.agent_workflows.workflow_runtime import RUNTIME_TEXT_FIELDS, SUPPORTED_RUNTIME_KINDS


BUILTIN_WORKFLOW_DIR = Path(__file__).with_name("builtins")

BUILTIN_DISCOVERY_CATEGORIES = {
    "router_rag_agent": "router",
    "plan_execute_rag_agent": "replanner",
    "evaluator_replanner_rag_agent": "replanner",
    "orchestrator_worker_rag_agent": "replanner",
    "corrective_self_rag_agent": "replanner",
    "deep_research_agent": "deep",
    "hermes_rag_agent": "deep",
}

HERMES_RUNTIME_KIND = "hermes_agent"


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
        runtime = spec_json.get("runtime")
        framework = str(payload.get("framework") or "").strip()
        if not framework:
            raise ValueError(f"Builtin workflow file {path} is missing framework identity")
        if framework == "langgraph":
            if not isinstance(runtime, dict) or runtime.get("kind") not in SUPPORTED_RUNTIME_KINDS:
                raise ValueError(f"Builtin workflow file {path} must contain supported spec_json.runtime")
            missing_runtime_fields = sorted(field for field in RUNTIME_TEXT_FIELDS if not isinstance(runtime.get(field), str) or not runtime.get(field))
            if missing_runtime_fields:
                raise ValueError(f"Builtin workflow file {path} runtime is missing: {', '.join(missing_runtime_fields)}")
        elif framework == "hermes":
            if not isinstance(runtime, dict) or runtime.get("kind") != HERMES_RUNTIME_KIND:
                raise ValueError(f"Builtin Hermes workflow file {path} must contain runtime.kind={HERMES_RUNTIME_KIND!r}")
        else:
            raise ValueError(f"Builtin workflow file {path} has unsupported framework {framework!r}")
        payload["framework"] = framework
        payload["builder_id"] = str(payload.get("builder_id") or "").strip()
        if not payload["builder_id"]:
            raise ValueError(f"Builtin workflow file {path} is missing builder identity")
        payload["category"] = str(payload.get("category") or BUILTIN_DISCOVERY_CATEGORIES.get(builtin_key) or "router")
        payloads.append(payload)
    return tuple(payloads)


def load_builtin_workflows() -> list[Dict[str, Any]]:
    return [deepcopy(payload) for payload in _builtin_workflow_payloads()]


def builtin_workflow_keys() -> frozenset[str]:
    return frozenset(str(payload["builtin_key"]) for payload in _builtin_workflow_payloads())
