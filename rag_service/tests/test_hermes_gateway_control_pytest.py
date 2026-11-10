import json
import socket
import threading
from pathlib import Path

import pytest

from hermes_runtime.gateway_control import ensure_profile_served, gateway_home, release_profile_served


def test_gateway_home_prefers_gateway_env(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HERMES_GATEWAY_HOME", str(tmp_path))
    assert gateway_home() == tmp_path


def test_ensure_profile_served_reads_serve_profile_response(monkeypatch, tmp_path: Path):
    sock_path = tmp_path / "gateway.sock"
    received: list[bytes] = []

    class Server:
        def __init__(self) -> None:
            self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server.bind(str(sock_path))
            self._server.listen(1)

        def handle_once(self) -> None:
            conn, _ = self._server.accept()
            with conn:
                data = b""
                while b"\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                received.append(data)
                payload = json.loads(data.decode("utf-8").strip())
                assert payload["verb"] == "serve-profile"
                conn.sendall(
                    json.dumps({"ok": True, "result": {"served_profiles": [payload["params"]["name"]]}}).encode()
                    + b"\n"
                )

        def close(self) -> None:
            self._server.close()

    monkeypatch.setenv("HERMES_GATEWAY_HOME", str(tmp_path))
    server = Server()
    done = threading.Event()

    def accept_once() -> None:
        server.handle_once()
        done.set()

    worker = threading.Thread(target=accept_once, daemon=True)
    worker.start()
    try:
        assert ensure_profile_served("askpdf-run-deadbeefdeadbeef") is True
    finally:
        done.wait(timeout=2)
        server.close()
        worker.join(timeout=2)
    assert received and b"serve-profile" in received[0]


def test_release_profile_served_confirms_profile_removed(monkeypatch, tmp_path: Path):
    sock_path = tmp_path / "gateway.sock"
    verbs: list[str] = []

    class Server:
        def __init__(self) -> None:
            self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server.bind(str(sock_path))
            self._server.listen(2)

        def handle_once(self) -> None:
            conn, _ = self._server.accept()
            with conn:
                data = b""
                while b"\n" not in data:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    data += chunk
                payload = json.loads(data.decode("utf-8").strip())
                verbs.append(str(payload["verb"]))
                if payload["verb"] == "unserve-profile":
                    conn.sendall(
                        json.dumps({"ok": True, "result": {"served_profiles": ["askpdf-deep-offline"]}}).encode()
                        + b"\n"
                    )
                else:
                    conn.sendall(
                        json.dumps({"ok": True, "result": {"served_profiles": ["askpdf-deep-offline"]}}).encode()
                        + b"\n"
                    )

        def close(self) -> None:
            self._server.close()

    monkeypatch.setenv("HERMES_GATEWAY_HOME", str(tmp_path))
    server = Server()
    responses = 2
    done = threading.Event()

    def accept_loop() -> None:
        for _ in range(responses):
            server.handle_once()
        done.set()

    worker = threading.Thread(target=accept_loop, daemon=True)
    worker.start()
    try:
        assert release_profile_served("askpdf-run-deadbeefdeadbeef") is True
    finally:
        done.wait(timeout=2)
        server.close()
        worker.join(timeout=2)
    assert "unserve-profile" in verbs
