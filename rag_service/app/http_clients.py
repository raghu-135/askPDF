"""Application-scoped HTTP client lifecycle."""

import asyncio
import math
import os
from typing import Any

import httpx

_clients: dict[str, httpx.AsyncClient] = {}
_owned: set[Any] = set()
_shared_lazy: set[Any] = set()


def _timeout(name: str = "default") -> httpx.Timeout:
    env_name = "MCP_REQUEST_TIMEOUT_SECONDS" if name == "mcp" else "HTTP_CLIENT_TIMEOUT_SECONDS"
    raw = os.getenv(env_name, "120" if name == "mcp" else "30").strip()
    try:
        seconds = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {env_name}={raw!r}; expected a finite positive number of seconds") from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(f"Invalid {env_name}={raw!r}; expected a finite positive number of seconds")
    return httpx.Timeout(seconds, connect=min(seconds, 10.0))


async def init_http_clients() -> None:
    limits = httpx.Limits(
        max_connections=int(os.getenv("HTTP_MAX_CONNECTIONS", "100")),
        max_keepalive_connections=int(os.getenv("HTTP_MAX_KEEPALIVE_CONNECTIONS", "20")),
    )
    for name in ("default", "llm", "embeddings", "providers", "mcp"):
        if name not in _clients:
            _clients[name] = httpx.AsyncClient(
                timeout=_timeout("mcp" if name == "mcp" else "default"),
                limits=limits,
            )


def get_http_client(name: str = "default") -> httpx.AsyncClient:
    client = _clients.get(name)
    if client is None:
        # Tests and CLI callers may use the client without the FastAPI lifespan.
        client = httpx.AsyncClient(timeout=_timeout(name))
        _clients[name] = client
        register_owned_client(client)
        _shared_lazy.add(client)
    return client


def register_owned_client(client: Any) -> Any:
    _owned.add(client)
    return client


def release_owned_client(client: Any) -> None:
    _owned.discard(client)


def is_owned_client(client: Any) -> bool:
    return client in _owned and client not in _shared_lazy


async def close_http_clients() -> None:
    clients = list({id(client): client for client in [*_clients.values(), *_owned]}.values())
    _clients.clear()
    _owned.clear()
    _shared_lazy.clear()
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Closing HTTP clients | count=%s", len(clients))
    for client in clients:
        close = getattr(client, "aclose", None)
        if close is not None:
            try:
                await close()
            except Exception:
                logger.exception("Failed to close HTTP client")
