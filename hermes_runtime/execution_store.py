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
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Protocol


_EXECUTION_GRANT_KEYS = frozenset({"mcp_execution_context_token", "_askpdf_context_token"})


def _without_execution_grants(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_execution_grants(item)
            for key, item in value.items()
            if str(key).lower() not in _EXECUTION_GRANT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_without_execution_grants(item) for item in value]
    return value


class HermesExecutionStoreProtocol(Protocol):
    """Storage contract for replacing proof storage with PostgreSQL later."""

    def create(self, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]: ...
    def update(self, run_id: str, **values: Any) -> None: ...
    def append(self, run_id: str, frame: str) -> bool: ...
    def finalize(self, run_id: str, frame: str, *, status: str) -> bool: ...
    def fail_in_memory(self, run_id: str, *, status: str = "failed") -> None: ...
    def probe(self) -> bool: ...
    def frames_after(self, run_id: str, after_event_id: str | None = None) -> list[str]: ...


class HermesExecutionConflictError(RuntimeError):
    """Raised when a run ID is reused for different execution semantics."""


class HermesStoreLoadError(RuntimeError):
    """Raised when an existing Hermes journal cannot be loaded safely."""


def request_fingerprint(payload: Mapping[str, Any]) -> str:
    """Fingerprint immutable start semantics while excluding transport metadata."""
    request = payload.get("request") or {}
    context = payload.get("context") or {}
    resolved_spec = context.get("resolved_spec") or {}
    value = {
        "operation": "start",
        "run_id": request.get("run_id"),
        "definition_id": request.get("definition_id"),
        "framework": request.get("framework"),
        "builder_id": request.get("builder_id"),
        "input": _without_execution_grants(request.get("input") or {}),
        "options": request.get("options") or {},
        "interrupt": request.get("interrupt") or {},
        "resolved_config": resolved_spec.get("config") or {},
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class HermesExecutionStore:
    _RECORD_KEYS = frozenset({
        "run_id",
        "status",
        "events",
        "payload",
        "request_fingerprint",
        "next_sequence",
        "last_event_id",
        "last_upstream_event_id",
        "terminal_event_id",
        "terminal_result",
        "continuation",
    })

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path or os.getenv("HERMES_RUNTIME_STATE_PATH", ""))
        self.records: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        try:
            exists = self.path.exists()
        except OSError as exc:
            raise HermesStoreLoadError(
                f"existing Hermes execution journal is unreadable: {self.path}"
            ) from exc
        if not exists:
            self.records = {}
            return
        try:
            value = json.loads(self.path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise HermesStoreLoadError(
                f"existing Hermes execution journal is unreadable: {self.path}"
            ) from exc
        if not isinstance(value, Mapping):
            raise HermesStoreLoadError("Hermes execution journal root must be an object")

        loaded: dict[str, dict[str, Any]] = {}
        for run_id, raw_record in value.items():
            if not isinstance(run_id, str) or not isinstance(raw_record, Mapping):
                raise HermesStoreLoadError("Hermes execution journal contains an invalid record")
            record = dict(raw_record)
            self._validate_record(run_id, record)
            loaded[run_id] = record

        self.records = loaded

    @classmethod
    def _validate_record(cls, run_id: str, record: Mapping[str, Any]) -> None:
        from runtime_protocol.protocol import decode_event_frame

        if set(record) != cls._RECORD_KEYS and set(record) != cls._RECORD_KEYS - {"continuation"}:
            raise HermesStoreLoadError(f"Hermes execution journal has an invalid record shape: {run_id}")
        if record.get("run_id") != run_id or not isinstance(record.get("status"), str):
            raise HermesStoreLoadError(f"Hermes execution journal has invalid record identity: {run_id}")
        if not isinstance(record.get("events"), list):
            raise HermesStoreLoadError(f"Hermes execution journal has invalid events for run: {run_id}")
        for event in record["events"]:
            if (
                not isinstance(event, Mapping)
                or set(event) != {"event_id", "frame"}
                or not isinstance(event.get("event_id"), str)
                or not isinstance(event.get("frame"), str)
            ):
                raise HermesStoreLoadError(f"Hermes execution journal has malformed event for run: {run_id}")
            try:
                canonical = decode_event_frame(event["frame"])["event"]
                if canonical["run_id"] != run_id or canonical["event_id"] != event["event_id"]:
                    raise ValueError("journal event identity mismatch")
            except (TypeError, ValueError, KeyError) as exc:
                raise HermesStoreLoadError(f"Hermes execution journal has invalid canonical event: {run_id}") from exc
        if not isinstance(record.get("payload"), Mapping):
            raise HermesStoreLoadError(f"Hermes execution journal has invalid payload for run: {run_id}")
        fingerprint = record.get("request_fingerprint")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise HermesStoreLoadError(f"Hermes execution journal has invalid request fingerprint: {run_id}")
        if not isinstance(record.get("next_sequence"), int) or record["next_sequence"] < 1:
            raise HermesStoreLoadError(f"Hermes execution journal has invalid sequence cursor: {run_id}")
        for key in ("last_event_id", "last_upstream_event_id", "terminal_event_id"):
            if record.get(key) is not None and not isinstance(record.get(key), str):
                raise HermesStoreLoadError(f"Hermes execution journal has invalid {key}: {run_id}")
        if record.get("terminal_result") is not None and not isinstance(record.get("terminal_result"), Mapping):
            raise HermesStoreLoadError(f"Hermes execution journal has invalid terminal result: {run_id}")
        if "continuation" in record and not isinstance(record["continuation"], Mapping):
            raise HermesStoreLoadError(f"Hermes execution journal has invalid continuation: {run_id}")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.records, separators=(",", ":"), default=str))
        temporary.chmod(0o600)
        temporary.replace(self.path)
        self.path.chmod(0o600)

    def create(self, run_id: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        fingerprint = request_fingerprint(payload)
        persisted_payload = _without_execution_grants(payload)
        existing = self.records.get(run_id)
        if existing is not None:
            existing_fingerprint = existing.get("request_fingerprint")
            if existing_fingerprint is None:
                stored_payload = existing.get("payload")
                if not isinstance(stored_payload, Mapping):
                    raise HermesExecutionConflictError(
                        "existing Hermes execution has no comparable request payload"
                    )
                existing_fingerprint = request_fingerprint(stored_payload)
                existing["request_fingerprint"] = existing_fingerprint
            if existing_fingerprint != fingerprint:
                raise HermesExecutionConflictError(
                    "run_id is already bound to different execution semantics"
                )
            return existing

        record = {
            "run_id": run_id,
            "status": "queued",
            "events": [],
            # Execution grants are short-lived bearer capabilities.  They are
            # intentionally excluded from the durable journal; recovery must
            # be re-admitted with a fresh grant by the control plane.
            "payload": persisted_payload,
            "request_fingerprint": fingerprint,
            "next_sequence": 1,
            "last_event_id": None,
            "last_upstream_event_id": None,
            "terminal_event_id": None,
            "terminal_result": None,
        }
        self.records[run_id] = record
        self._save()
        return record

    def _record(self, run_id: str) -> dict[str, Any]:
        record = self.records.get(run_id)
        if record is None:
            raise HermesStoreLoadError(f"Hermes execution journal has no record for run: {run_id}")
        return record

    def update(self, run_id: str, **values: Any) -> None:
        record = self._record(run_id)
        record.update(values)
        self._save()

    def next_sequence(self, run_id: str) -> int:
        record = self._record(run_id)
        value = record.get("next_sequence")
        if type(value) is not int or value < 1:
            raise HermesStoreLoadError(f"Hermes execution journal has invalid sequence cursor: {run_id}")
        return value

    def _append_frame(self, run_id: str, frame: str) -> bool:
        from runtime_protocol.protocol import decode_event_frame

        body = decode_event_frame(frame)
        event = body["event"]
        if event["run_id"] != run_id:
            raise ValueError("event belongs to a different Hermes run")
        record = self._record(run_id)
        events = record["events"]
        event_id = event["event_id"]
        source_event_id = event.get("source_event_id")
        for item in events:
            prior = decode_event_frame(item["frame"])["event"]
            if item["event_id"] == event_id or (source_event_id and prior.get("source_event_id") == source_event_id):
                return False
        if events:
            previous = decode_event_frame(events[-1]["frame"])["event"]
            if previous["terminal"] or event["sequence"] <= previous["sequence"]:
                raise ValueError("Hermes events must be ordered and cannot follow a terminal event")
        events.append({"event_id": event_id, "frame": frame})
        if event.get("continuation") is not None:
            record["continuation"] = event["continuation"]
        record["last_event_id"] = event_id
        record["next_sequence"] = event["sequence"] + 1
        if source_event_id:
            record["last_upstream_event_id"] = source_event_id
        if event["terminal"]:
            record["terminal_event_id"] = event_id
            if body.get("result") is not None:
                record["terminal_result"] = body["result"]
        return True

    def append(self, run_id: str, frame: str) -> bool:
        appended = self._append_frame(run_id, frame)
        if appended:
            self._save()
        return appended

    def finalize(self, run_id: str, frame: str, *, status: str) -> bool:
        """Persist one terminal frame and status with a single durable save."""
        record = self._record(run_id)
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
        record = self._record(run_id)
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
