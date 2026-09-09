from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any, Mapping

import httpx

from langgraph_runtime.limits import required_positive_float
from runtime_protocol.configuration import parse_bounded_ratio
from langgraph_runtime.models.llm import provider_configuration


logger = logging.getLogger(__name__)


def _timestamp(value: float | None = None) -> str:
    return datetime.fromtimestamp(value or time.time(), timezone.utc).isoformat()


async def probe_mcp(url: str, timeout: float, *, client: httpx.AsyncClient | None = None) -> dict[str, Any]:
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        response = await client.post(
            url,
            headers={"accept": "application/json, text/event-stream"},
            json={"jsonrpc": "2.0", "id": "runtime-dependency-check", "method": "tools/list", "params": {}},
        )
        if not 200 <= response.status_code < 300:
            return {"ok": False, "reason": "unexpected_status", "http_status": response.status_code}
        try:
            payload = response.json()
        except ValueError:
            return {"ok": False, "reason": "invalid_json", "http_status": response.status_code}
        result = payload.get("result") if isinstance(payload, Mapping) else None
        tools = result.get("tools") if isinstance(result, Mapping) else None
        if not isinstance(payload, Mapping) or payload.get("error"):
            return {"ok": False, "reason": "mcp_error", "http_status": response.status_code}
        if not isinstance(tools, list):
            return {"ok": False, "reason": "invalid_tools_list", "http_status": response.status_code}
        tool_ids: set[str] = set()
        for item in tools:
            if not isinstance(item, Mapping):
                continue
            if item.get("name"):
                tool_ids.add(str(item["name"]))
            metadata = item.get("_meta") if isinstance(item.get("_meta"), Mapping) else {}
            if metadata.get("com.askpdf/contract-id"):
                tool_ids.add(str(metadata["com.askpdf/contract-id"]))
        return {"ok": True, "http_status": response.status_code, "capability_ids": sorted(tool_ids), "protocol": "mcp"}
    except Exception as exc:
        return {"ok": False, "reason": type(exc).__name__}
    finally:
        if owns_client:
            await client.aclose()


async def probe_provider(url: str, timeout: float, *, client: httpx.AsyncClient | None = None) -> dict[str, Any]:
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout)
    try:
        provider_base, headers, _api_key = provider_configuration(url)
        models_url = provider_base + "/models" if provider_base.endswith("/v1") else provider_base + "/v1/models"
        response = await client.get(models_url, headers=headers)
        if not 200 <= response.status_code < 300:
            return {"ok": False, "reason": "unexpected_status", "http_status": response.status_code}
        try:
            payload = response.json()
        except ValueError:
            return {"ok": False, "reason": "invalid_json", "http_status": response.status_code}
        models = payload.get("data") if isinstance(payload, Mapping) else None
        if not isinstance(models, list):
            return {"ok": False, "reason": "invalid_models_list", "http_status": response.status_code}
        model_ids = sorted({str(item.get("id")) for item in models if isinstance(item, Mapping) and item.get("id")})
        return {"ok": True, "http_status": response.status_code, "capability_ids": model_ids}
    except Exception as exc:
        return {"ok": False, "reason": type(exc).__name__}
    finally:
        if owns_client:
            await client.aclose()


class DependencyMonitor:
    def __init__(self) -> None:
        self.interval = required_positive_float("AGENT_RUNTIME_DEPENDENCY_REFRESH_SECONDS")
        self.timeout = required_positive_float("AGENT_RUNTIME_DEPENDENCY_TIMEOUT_SECONDS")
        self.stale_after = max(self.interval, required_positive_float("AGENT_RUNTIME_DEPENDENCY_STALE_SECONDS"))
        try:
            self.jitter = parse_bounded_ratio(
                os.getenv("AGENT_RUNTIME_DEPENDENCY_JITTER_RATIO", ""),
                name="AGENT_RUNTIME_DEPENDENCY_JITTER_RATIO",
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        self._configured = {
            "mcp": os.getenv("MCP_LOOPBACK_URL", "").strip(),
            "provider": os.getenv("LLM_API_URL", "").strip(),
        }
        self._snapshots: dict[str, dict[str, Any]] = {
            name: {"state": "not_configured" if not url else "unavailable", "checked_at": None, "last_success_at": None, "reason": "not_configured" if not url else "not_checked", "capability_ids": []}
            for name, url in self._configured.items()
        }
        self.counters = {"checks": 0, "failures": 0, "transitions": 0}

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        current = now or time.time()
        result: dict[str, Any] = {}
        for name, value in self._snapshots.items():
            item = dict(value)
            last_success_epoch = item.pop("_last_success_epoch", None)
            if item["state"] == "available" and last_success_epoch and current - last_success_epoch > self.stale_after:
                item["state"] = "degraded"
                item["reason"] = "stale"
            result[name] = item
        return result

    async def refresh(self) -> None:
        for name, url in self._configured.items():
            if not url:
                continue
            started = time.monotonic()
            probe = probe_mcp if name == "mcp" else probe_provider
            value = await probe(url, self.timeout)
            self.counters["checks"] += 1
            previous = self._snapshots[name]
            now = time.time()
            if value.get("ok"):
                updated = {
                    "state": "available",
                    "checked_at": _timestamp(now),
                    "last_success_at": _timestamp(now),
                    "reason": None,
                    "capability_ids": list(value.get("capability_ids") or []),
                    "latency_ms": round((time.monotonic() - started) * 1000, 1),
                    "_last_success_epoch": now,
                }
                if value.get("protocol"):
                    updated["protocol"] = value["protocol"]
            else:
                self.counters["failures"] += 1
                updated = dict(previous)
                updated["checked_at"] = _timestamp(now)
                updated["reason"] = str(value.get("reason") or "probe_failed")
                updated["latency_ms"] = round((time.monotonic() - started) * 1000, 1)
                if not previous.get("_last_success_epoch"):
                    updated["state"] = "unavailable"
                elif now - float(previous["_last_success_epoch"]) > self.stale_after:
                    updated["state"] = "degraded"
            if previous.get("state") != updated.get("state"):
                self.counters["transitions"] += 1
                logger.info("Runtime dependency state changed | dependency=%s state=%s reason=%s", name, updated["state"], updated.get("reason"))
            self._snapshots[name] = updated

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            delay = self.interval * (1 + random.uniform(-self.jitter, self.jitter))
            try:
                await asyncio.wait_for(stop.wait(), timeout=max(1.0, delay))
                continue
            except asyncio.TimeoutError:
                await self.refresh()

    def unavailable(self, requirements: Mapping[str, set[str]]) -> dict[str, Any] | None:
        snapshots = self.snapshot()
        for dependency, required_ids in requirements.items():
            if not required_ids:
                continue
            value = snapshots.get(dependency) or {"state": "not_configured", "capability_ids": []}
            available_ids = set(value.get("capability_ids") or [])
            missing = sorted(required_ids - available_ids)
            if value.get("state") != "available" or missing:
                return {
                    "dependency": dependency,
                    "reason": value.get("reason") or ("missing_capabilities" if missing else "unavailable"),
                    "missing_capability_ids": missing,
                }
        return None

    def unavailable_configured(self) -> dict[str, Any] | None:
        """Return the first configured dependency that cannot accept work."""

        snapshots = self.snapshot()
        for dependency, url in self._configured.items():
            if not url:
                continue
            value = snapshots.get(dependency) or {}
            if value.get("state") != "available":
                return {
                    "dependency": dependency,
                    "reason": value.get("reason") or "unavailable",
                }
        return None


def langgraph_dependency_requirements(payload: Mapping[str, Any]) -> dict[str, set[str]]:
    request = payload.get("request") if isinstance(payload.get("request"), Mapping) else {}
    request_input = request.get("input") if isinstance(request.get("input"), Mapping) else {}
    context = payload.get("context") if isinstance(payload.get("context"), Mapping) else {}
    # The control plane expands neutral workflow contracts into the canonical
    # MCP names authorized by the signed execution token. Do not admit against
    # resolved_spec.allowed_tool_ids: that list can also contain graph-local
    # capabilities which an MCP server must not advertise.
    tools = {
        str(item)
        for item in request_input.get("mcp_allowed_tool_ids") or []
        if isinstance(item, str) and item
    }
    options = request.get("options") if isinstance(request.get("options"), Mapping) else {}
    request_payload = context.get("request_payload") if isinstance(context.get("request_payload"), Mapping) else {}
    task_context = context.get("task_context") if isinstance(context.get("task_context"), Mapping) else {}
    task_metadata = task_context.get("metadata") if isinstance(task_context.get("metadata"), Mapping) else {}
    # Embeddings are resolved by the control-plane MCP tools.  They are
    # carried in the execution context so those tools can use the thread's
    # configured embedding model, but they are not models required from the
    # LangGraph runtime's chat provider.
    models = {
        str(value)
        for value in (
            options.get("llm_model") or request_payload.get("llm_model") or task_metadata.get("llm_model"),
        )
        if value
    }
    return {"mcp": tools, "provider": models}
