"""Small durable gateway journal for Hermes executions.

The upstream Hermes run/session identifiers and event frames are persisted so
the gateway can reconnect subscribers after a process restart.  The file is
intended to be backed by a persistent container volume; deployments may swap
this implementation for the same Postgres schema used by the LangGraph
runtime without changing the gateway API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


class HermesExecutionStore:
    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or os.getenv("HERMES_RUNTIME_STATE_PATH", "/tmp/askpdf-hermes-runtime.json"))
        self.records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            value = json.loads(self.path.read_text())
            self.records = dict(value) if isinstance(value, Mapping) else {}
        except (FileNotFoundError, OSError, ValueError):
            self.records = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.records, separators=(",", ":"), default=str))
        temporary.replace(self.path)

    def create(self, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = self.records.setdefault(run_id, {"run_id": run_id, "status": "queued", "events": [], "payload": dict(payload)})
        self._save()
        return record

    def update(self, run_id: str, **values: Any) -> None:
        record = self.records.setdefault(run_id, {"run_id": run_id, "events": []})
        record.update(values)
        self._save()

    def append(self, run_id: str, frame: str) -> None:
        record = self.records.setdefault(run_id, {"run_id": run_id, "events": []})
        events = record.setdefault("events", [])
        event_id = next((line[3:].strip() for line in frame.splitlines() if line.startswith("id:")), f"{run_id}:{len(events) + 1}")
        if any(item.get("event_id") == event_id for item in events):
            return
        events.append({"event_id": event_id, "frame": frame})
        try:
            data = next((line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:")), None)
            payload = json.loads(data) if data else {}
            event = payload.get("event") if isinstance(payload, Mapping) else None
            continuation = event.get("continuation") if isinstance(event, Mapping) else None
            if continuation:
                record["continuation"] = continuation
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        self._save()

    def frames_after(self, run_id: str, after_event_id: str | None = None) -> list[str]:
        events = self.records.get(run_id, {}).get("events", [])
        if not after_event_id:
            return [str(item.get("frame", "")) for item in events]
        for index, item in enumerate(events):
            if item.get("event_id") == after_event_id:
                return [str(value.get("frame", "")) for value in events[index + 1:]]
        return [str(item.get("frame", "")) for item in events]
