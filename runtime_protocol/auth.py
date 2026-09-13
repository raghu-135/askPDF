"""Dependency-light authentication helpers shared by external runtimes."""

from __future__ import annotations

import hmac


PUBLIC_OPERATIONAL_PATHS = frozenset({"/healthz", "/startupz", "/readyz"})


def bearer_token(authorization: str | None) -> str | None:
    """Parse the one accepted service-auth wire shape."""

    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[len("Bearer "):]
    if not token or token != token.strip() or any(character.isspace() for character in token):
        return None
    return token


def valid_bearer_token(authorization: str | None, expected: str) -> bool:
    supplied = bearer_token(authorization)
    return supplied is not None and hmac.compare_digest(supplied, expected)
