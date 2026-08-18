"""Single-process proof journal for Hermes executions.

The upstream Hermes run/session identifiers and event frames are persisted so
the gateway can reconnect subscribers after a process restart.  The file is
intended to be backed by a persistent container volume; deployments may swap
this implementation for the same Postgres schema used by the LangGraph
runtime without changing the gateway API. This implementation is not safe for
multiple workers or replicas.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol


class HermesExecutionStoreProtocol(Protocol):
    """Storage contract for replacing proof storage with PostgreSQL later."""

    def create(self, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]: ...
    def update(self, run_id: str, **values: Any) -> None: ...
    def append(self, run_id: str, frame: str) -> bool: ...
    def finalize(self, run_id: str, frame: str, *, status: str) -> bool: ...
    def fail_in_memory(self, run_id: str, *, status: str = "failed") -> None: ...
    def probe(self) -> bool: ...
    def frames_after(self, run_id: str, after_event_id: str | None = None) -> list[str]: ...


class HermesExecutionStore:
    SCHEMA_VERSION = 2

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or os.getenv("HERMES_RUNTIME_STATE_PATH", "/tmp/askpdf-hermes-runtime.json"))
        self.records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            value = json.loads(self.path.read_text())
            self.records = dict(value) if isinstance(value, Mapping) else {}
            for record in self.records.values():
                if not isinstance(record, Mapping):
                    continue
                events = record.get("events") or []
                sequences = []
                for item in events:
                    event_id = str(item.get("event_id") or "") if isinstance(item, Mapping) else ""
                    try:
                        sequences.append(int(event_id.rsplit(":", 1)[-1]))
                    except ValueError:
                        pass
                record.setdefault("event_schema_version", self.SCHEMA_VERSION)
                record.setdefault("next_sequence", max(sequences, default=len(events)) + 1)
                record.setdefault("last_event_id", events[-1].get("event_id") if events else None)
                record.setdefault("last_upstream_event_id", None)
                record.setdefault("terminal_event_id", None)
                record.setdefault("terminal_result", None)
        except (FileNotFoundError, OSError, ValueError):
            self.records = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.records, separators=(",", ":"), default=str))
        temporary.replace(self.path)

    def create(self, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        record = self.records.setdefault(run_id, {"run_id": run_id, "status": "queued", "events": [], "payload": dict(payload)})
        record.setdefault("event_schema_version", self.SCHEMA_VERSION)
        record.setdefault("next_sequence", 1)
        record.setdefault("last_event_id", None)
        record.setdefault("last_upstream_event_id", None)
        record.setdefault("terminal_event_id", None)
        record.setdefault("terminal_result", None)
        self._save()
        return record

    def update(self, run_id: str, **values: Any) -> None:
        record = self.records.setdefault(run_id, {"run_id": run_id, "events": []})
        record.update(values)
        self._save()

    def next_sequence(self, run_id: str) -> int:
        record = self.records.setdefault(run_id, {"run_id": run_id, "events": []})
        value = record.get("next_sequence")
        if isinstance(value, int) and value > 0:
            return value
        events = record.get("events") or []
        value = len(events) + 1
        record["next_sequence"] = value
        return value

    def _append_frame(self, run_id: str, frame: str) -> bool:
        record = self.records.setdefault(run_id, {"run_id": run_id, "events": []})
        events = record.setdefault("events", [])
        event_id = next((line[3:].strip() for line in frame.splitlines() if line.startswith("id:")), f"{run_id}:{len(events) + 1}")
        data = next((line[5:].lstrip() for line in frame.splitlines() if line.startswith("data:")), None)
        try:
            decoded = json.loads(data) if data else {}
            decoded_event = decoded.get("event") if isinstance(decoded, Mapping) else None
            source_event_id = decoded_event.get("source_event_id") if isinstance(decoded_event, Mapping) else None
        except (TypeError, ValueError, json.JSONDecodeError):
            decoded = {}
            source_event_id = None
        if any(item.get("event_id") == event_id for item in events):
            return False
        if source_event_id:
            for item in events:
                try:
                    prior_data = next(line[5:].lstrip() for line in str(item.get("frame", "")).splitlines() if line.startswith("data:"))
                    prior = json.loads(prior_data).get("event") or {}
                    if prior.get("source_event_id") == source_event_id:
                        return False
                except (StopIteration, TypeError, ValueError, json.JSONDecodeError):
                    continue
        events.append({"event_id": event_id, "frame": frame})
        try:
            payload = json.loads(data) if data else {}
            event = payload.get("event") if isinstance(payload, Mapping) else None
            continuation = event.get("continuation") if isinstance(event, Mapping) else None
            if continuation:
                record["continuation"] = continuation
            event = event if isinstance(event, Mapping) else {}
            record["last_event_id"] = event_id
            record["next_sequence"] = max(self.next_sequence(run_id), int(event.get("sequence") or 0) + 1)
            source_event_id = event.get("source_event_id")
            if source_event_id:
                record["last_upstream_event_id"] = str(source_event_id)
            if event.get("terminal"):
                record["terminal_event_id"] = event_id
                if payload.get("result") is not None:
                    record["terminal_result"] = payload.get("result")
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        return True

    def append(self, run_id: str, frame: str) -> bool:
        appended = self._append_frame(run_id, frame)
        if appended:
            self._save()
        return appended

    def finalize(self, run_id: str, frame: str, *, status: str) -> bool:
        """Persist one terminal frame and status with a single durable save."""
        record = self.records.setdefault(run_id, {"run_id": run_id, "events": []})
        if record.get("status") in {"completed", "failed", "cancelled"} or record.get("terminal_event_id"):
            return False
        snapshot = copy.deepcopy(record)
        appended = self._append_frame(run_id, frame)
        record["status"] = status
        try:
            self._save()
        except Exception:
            self.records[run_id] = snapshot
            raise
        return appended

    def fail_in_memory(self, run_id: str, frame: str) -> None:
        """Terminate live subscribers when durable storage is unavailable."""
        record = self.records.setdefault(run_id, {"run_id": run_id, "events": []})
        if not record.get("terminal_event_id"):
            self._append_frame(run_id, frame)
        record["status"] = "failed"

    def probe(self) -> bool:
        """Verify that the current journal can be durably written."""
        try:
            self._save()
            return True
        except Exception:
            return False

    def frames_after(self, run_id: str, after_event_id: str | None = None) -> list[str]:
        events = self.records.get(run_id, {}).get("events", [])
        if not after_event_id:
            return [str(item.get("frame", "")) for item in events]
        for index, item in enumerate(events):
            if item.get("event_id") == after_event_id:
                return [str(value.get("frame", "")) for value in events[index + 1:]]
        return [str(item.get("frame", "")) for item in events]
