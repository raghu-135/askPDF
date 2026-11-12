#!/usr/bin/env python3
"""Minimal OpenAI-compatible reverse proxy on the VM host for Compose containers.

Containers reach this via docker-compose.vm-hostgateway.override.yml at
http://host.docker.internal:12345/v1 . The proxy forwards to OpenRouter (or any
OpenAI-compatible upstream) using credentials from the host environment only.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_UPSTREAM = "https://openrouter.ai/api/v1"
DEFAULT_PORT = 12345
HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


def _upstream_base() -> str:
    base = os.environ.get("OPENROUTER_UPSTREAM", DEFAULT_UPSTREAM).rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return base


def _api_key() -> str:
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


class _ProxyHandler(BaseHTTPRequestHandler):
    server_version = "askpdf-openrouter-proxy/1.0"

    def log_message(self, fmt: str, *args) -> None:
        if os.environ.get("OPENROUTER_PROXY_QUIET", "1") == "1":
            return
        super().log_message(fmt, *args)

    def do_GET(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_PUT(self) -> None:
        self._proxy()

    def do_PATCH(self) -> None:
        self._proxy()

    def do_DELETE(self) -> None:
        self._proxy()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.end_headers()

    def _proxy(self) -> None:
        api_key = _api_key()
        if not api_key:
            self.send_error(503, "OPENROUTER_API_KEY or OPENAI_API_KEY is not set on the host")
            return

        path = self.path
        if path.startswith("/v1"):
            path = path[3:]
        if not path.startswith("/"):
            path = "/" + path
        target = f"{_upstream_base()}{path}"

        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else None

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": self.headers.get("Content-Type", "application/json"),
            "Accept": self.headers.get("Accept", "application/json"),
        }
        for header in ("HTTP-Referer", "X-Title"):
            if self.headers.get(header):
                headers[header] = self.headers[header]

        request = urllib.request.Request(target, data=body, headers=headers, method=self.command)
        try:
            with urllib.request.urlopen(request, timeout=600) as upstream:
                payload = upstream.read()
                self.send_response(upstream.status)
                for key, value in upstream.headers.items():
                    lowered = key.lower()
                    if lowered not in HOP_BY_HOP:
                        self.send_header(key, value)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                if payload:
                    self.wfile.write(payload)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            for key, value in exc.headers.items():
                lowered = key.lower()
                if lowered not in HOP_BY_HOP:
                    self.send_header(key, value)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if payload:
                self.wfile.write(payload)
        except urllib.error.URLError as exc:
            detail = json.dumps({"error": {"message": str(exc.reason), "type": "proxy_error"}}).encode()
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(detail)))
            self.end_headers()
            self.wfile.write(detail)


def main() -> int:
    port = int(os.environ.get("OPENROUTER_PROXY_PORT", DEFAULT_PORT))
    bind = os.environ.get("OPENROUTER_PROXY_BIND", "0.0.0.0")
    if not _api_key():
        print("openrouter_host_proxy: OPENROUTER_API_KEY or OPENAI_API_KEY required", file=sys.stderr)
        return 1
    server = ThreadingHTTPServer((bind, port), _ProxyHandler)
    print(f"openrouter_host_proxy: listening on {bind}:{port} -> {_upstream_base()}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
