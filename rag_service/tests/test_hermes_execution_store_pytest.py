from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_runtime.execution_store import (
    HermesExecutionConflictError,
    HermesExecutionStore,
    HermesStoreLoadError,
    request_fingerprint,
)
from hermes_test_helpers import runtime_payload


def _gateway_frame(
    run_id: str,
    sequence: int,
    kind: str,
    *,
    source_event_id: str | None = None,
    terminal: bool = False,
    result: dict | None = None,
) -> str:
    event = {
        "event_id": f"{run_id}:{sequence}",
        "run_id": run_id,
        "sequence": sequence,
        "kind": kind,
        "payload": {},
        "terminal": terminal,
    }
    if source_event_id is not None:
        event["source_event_id"] = source_event_id
    body = {"event": event}
    if result is not None:
        body["result"] = result
    return f"id: {event['event_id']}\nevent: {kind}\ndata: {json.dumps(body)}\n\n"


def test_missing_hermes_store_is_a_valid_empty_store(tmp_path: Path) -> None:
    store = HermesExecutionStore(str(tmp_path / "missing.json"))
    assert store.records == {}


@pytest.mark.parametrize("content", ["{not-json", "[]", '{"run-1": "invalid"}'])
def test_existing_malformed_hermes_store_fails_closed(tmp_path: Path, content: str) -> None:
    path = tmp_path / "hermes.json"
    path.write_text(content)
    with pytest.raises(HermesStoreLoadError):
        HermesExecutionStore(str(path))


def test_existing_unreadable_hermes_store_fails_closed(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "hermes.json"
    path.write_text("{}")
    original = Path.read_text

    def unreadable(candidate: Path, *args, **kwargs):
        if candidate == path:
            raise OSError("permission denied")
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    with pytest.raises(HermesStoreLoadError, match="unreadable"):
        HermesExecutionStore(str(path))


def test_legacy_record_is_migrated_with_request_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "hermes.json"
    payload = runtime_payload("legacy-run")
    path.write_text(
        json.dumps(
            {
                "legacy-run": {
                    "run_id": "legacy-run",
                    "status": "completed",
                    "events": [],
                    "payload": payload,
                }
            }
        )
    )

    store = HermesExecutionStore(str(path))

    assert store.records["legacy-run"]["request_fingerprint"] == request_fingerprint(payload)
    persisted = json.loads(path.read_text())
    assert persisted["legacy-run"]["request_fingerprint"] == request_fingerprint(payload)


def test_reloaded_store_replays_identical_start_and_rejects_conflict(tmp_path: Path) -> None:
    path = tmp_path / "hermes.json"
    original = runtime_payload("fingerprinted-run")
    store = HermesExecutionStore(str(path))
    created = store.create("fingerprinted-run", original)

    reloaded = HermesExecutionStore(str(path))
    assert (
        reloaded.create("fingerprinted-run", original)["request_fingerprint"]
        == created["request_fingerprint"]
    )
    conflicting = json.loads(json.dumps(original))
    conflicting["request"]["options"]["llm_model"] = "different-model"
    with pytest.raises(HermesExecutionConflictError):
        reloaded.create("fingerprinted-run", conflicting)


def test_hermes_store_preserves_sequence_and_terminal_result_across_reload(tmp_path: Path) -> None:
    path = tmp_path / "hermes.json"
    store = HermesExecutionStore(str(path))
    store.create("run-sequence", {"request": {"run_id": "run-sequence"}})
    store.append("run-sequence", _gateway_frame("run-sequence", 1, "output.delta", source_event_id="upstream-1"))
    store.append("run-sequence", _gateway_frame("run-sequence", 2, "output.delta", source_event_id="upstream-2"))
    store.append(
        "run-sequence",
        _gateway_frame(
            "run-sequence",
            3,
            "run.completed",
            source_event_id="upstream-3",
            terminal=True,
            result={"status": "completed", "output": "ok"},
        ),
    )

    reloaded = HermesExecutionStore(str(path))
    record = reloaded.records["run-sequence"]
    assert record["next_sequence"] == 4
    assert record["terminal_event_id"] == "run-sequence:3"
    assert record["terminal_result"]["output"] == "ok"
    assert (
        reloaded.append(
            "run-sequence",
            _gateway_frame(
                "run-sequence",
                4,
                "run.completed",
                source_event_id="upstream-3",
                terminal=True,
                result={"status": "completed", "output": "ok"},
            ),
        )
        is False
    )


def test_hermes_store_finalizes_terminal_frame_and_status_in_one_save(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "hermes.json"
    store = HermesExecutionStore(str(path))
    store.create("run-terminal", {"request": {"run_id": "run-terminal"}})
    store.update("run-terminal", status="running")
    saves = 0
    original_save = store._save

    def counted_save() -> None:
        nonlocal saves
        saves += 1
        original_save()

    monkeypatch.setattr(store, "_save", counted_save)
    frame = _gateway_frame("run-terminal", 1, "run.failed", terminal=True, result={"status": "failed"})
    assert store.finalize("run-terminal", frame, status="failed") is True
    assert saves == 1
    reloaded = HermesExecutionStore(str(path))
    assert reloaded.records["run-terminal"]["status"] == "failed"
    assert reloaded.records["run-terminal"]["terminal_event_id"] == "run-terminal:1"


def test_hermes_store_failed_finalize_rolls_back_to_nonterminal_record(tmp_path: Path, monkeypatch) -> None:
    store = HermesExecutionStore(str(tmp_path / "hermes.json"))
    store.create("run-write-failure", {"request": {"run_id": "run-write-failure"}})
    store.update("run-write-failure", status="running")

    def fail_save() -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(store, "_save", fail_save)
    frame = _gateway_frame("run-write-failure", 1, "run.failed", terminal=True, result={"status": "failed"})
    with pytest.raises(OSError, match="disk unavailable"):
        store.finalize("run-write-failure", frame, status="failed")
    assert store.records["run-write-failure"]["status"] == "running"
    assert store.records["run-write-failure"].get("terminal_event_id") is None
