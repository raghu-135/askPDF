"""SSRF-safe URL checks for in-container browser navigation."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class BrowserUrlPolicyError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def validate_navigation_url(url: str) -> str:
    normalized = str(url or "").strip()
    if not normalized:
        raise BrowserUrlPolicyError(
            "browser_invalid_url",
            "A non-empty http(s) URL is required.",
        )
    if normalized.casefold().startswith("about:"):
        raise BrowserUrlPolicyError(
            "browser_capture_blank_page",
            "Internal about: pages cannot be opened.",
        )
    parsed = urlparse(normalized)
    if parsed.scheme not in {"http", "https"}:
        raise BrowserUrlPolicyError(
            "browser_invalid_url",
            "Only http:// and https:// URLs are allowed.",
        )
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise BrowserUrlPolicyError("browser_invalid_url", "URL must include a host.")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost"):
        raise BrowserUrlPolicyError(
            "browser_url_blocked",
            "Navigation to local hostnames is blocked.",
        )
    if host in {"metadata.google.internal", "metadata"}:
        raise BrowserUrlPolicyError("browser_url_blocked", "Navigation to metadata endpoints is blocked.")
    literal = host.strip("[]")
    try:
        address = ipaddress.ip_address(literal)
    except ValueError:
        address = None
    if address is not None:
        if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
            raise BrowserUrlPolicyError(
                "browser_url_blocked",
                "Navigation to private or reserved addresses is blocked.",
            )
        return normalized
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise BrowserUrlPolicyError(
            "browser_invalid_url",
            f"Could not resolve host: {host}",
        ) from exc
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        resolved = ipaddress.ip_address(sockaddr[0])
        if resolved.is_private or resolved.is_loopback or resolved.is_link_local or resolved.is_reserved:
            raise BrowserUrlPolicyError(
                "browser_url_blocked",
                "Navigation resolves to a private or reserved address.",
            )
    return normalized