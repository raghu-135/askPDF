#!/usr/bin/env python3
"""Fill local service-secret placeholders in .env.

Startup rejects documented placeholders such as replace-with-.... This writes a
fresh 64-character hex value for each matching assignment. It does not invent
provider credentials (OPENAI_API_KEY, HF_TOKEN) and does not overwrite secrets
that are already set.

The Compose env-secrets job also exports the service secrets to a shared
volume so app containers can load them in the same `docker compose up`.
"""

from __future__ import annotations

import argparse
import os
import re
import secrets
import shutil
import sys
from pathlib import Path

_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_PLACEHOLDER_PREFIXES = ("replace-with-", "change-me", "changeme")
_PROVIDER_CREDENTIALS = frozenset({"OPENAI_API_KEY", "HF_TOKEN"})
_HEX_BYTES = 32
SERVICE_SECRET_NAMES = (
    "LANGGRAPH_RUNTIME_TOKEN",
    "LANGGRAPH_RUNTIME_BINDING_SECRET",
    "MCP_EXECUTION_CONTEXT_SECRET",
    "ASKPDF_ADMIN_TOKEN",
    "HERMES_RUNTIME_TOKEN",
    "HERMES_API_TOKEN",
)


def _is_placeholder(value: str) -> bool:
    lowered = value.strip().lower()
    return any(lowered.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES)


def parse_env_assignments(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        match = _ASSIGNMENT.fullmatch(line.rstrip("\r"))
        if match is not None:
            values[match.group(1)] = match.group(2)
    return values


def service_secrets_from_values(values: dict[str, str]) -> dict[str, str] | None:
    secrets_map: dict[str, str] = {}
    for name in SERVICE_SECRET_NAMES:
        value = values.get(name, "").strip()
        if not value or _is_placeholder(value):
            return None
        secrets_map[name] = value
    return secrets_map


def write_runtime_secrets_file(secrets_map: dict[str, str], path: Path) -> None:
    missing = [name for name in SERVICE_SECRET_NAMES if name not in secrets_map]
    if missing:
        raise RuntimeError("missing service secrets: " + ", ".join(missing))
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{name}={secrets_map[name]}\n" for name in SERVICE_SECRET_NAMES)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(body)
    tmp.replace(path)


def fill_env_text(text: str, *, used: set[str] | None = None) -> tuple[str, list[str]]:
    """Return updated .env contents and the names that received new secrets."""
    occupied = set(used or ())
    filled: list[str] = []
    lines: list[str] = []
    for line in text.splitlines(keepends=True):
        newline = ""
        body = line
        if line.endswith("\r\n"):
            newline = "\r\n"
            body = line[:-2]
        elif line.endswith("\n"):
            newline = "\n"
            body = line[:-1]
        match = _ASSIGNMENT.fullmatch(body)
        if match is None or match.group(1) in _PROVIDER_CREDENTIALS or not _is_placeholder(match.group(2)):
            lines.append(line)
            continue
        secret = secrets.token_hex(_HEX_BYTES)
        while secret in occupied:
            secret = secrets.token_hex(_HEX_BYTES)
        occupied.add(secret)
        filled.append(match.group(1))
        lines.append(f"{match.group(1)}={secret}{newline}")
    return "".join(lines), filled


def fill_env_file(env_path: Path, *, example_path: Path | None = None) -> list[str]:
    if not env_path.exists():
        if example_path is None or not example_path.exists():
            raise FileNotFoundError(f"{env_path} does not exist and no example file was provided")
        shutil.copyfile(example_path, env_path)
    original = env_path.read_text()
    updated, filled = fill_env_text(original)
    if filled:
        env_path.write_text(updated)
    return filled


def resolve_service_secrets(
    env_path: Path,
    *,
    example_path: Path | None,
    environ: dict[str, str],
) -> tuple[dict[str, str], list[str], str]:
    """Return service secrets, names filled in .env, and the source used."""
    environ_secrets = service_secrets_from_values(environ)
    if env_path.exists():
        filled = fill_env_file(env_path, example_path=example_path)
        secrets_map = service_secrets_from_values(parse_env_assignments(env_path.read_text()))
        if secrets_map is None:
            raise RuntimeError(f"service secrets in {env_path} are missing or still placeholders")
        return secrets_map, filled, "file"
    if environ_secrets is not None:
        return environ_secrets, [], "environ"
    filled = fill_env_file(env_path, example_path=example_path)
    secrets_map = service_secrets_from_values(parse_env_assignments(env_path.read_text()))
    if secrets_map is None:
        raise RuntimeError(f"service secrets in {env_path} are missing or still placeholders")
    return secrets_map, filled, "file"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        type=Path,
        default=Path(".env"),
        help="Destination env file (default: .env)",
    )
    parser.add_argument(
        "--example",
        type=Path,
        default=Path(".env.example"),
        help="Copied to --env when that file is missing (default: .env.example)",
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Write KEY=value service secrets for Compose app containers",
    )
    parser.add_argument(
        "--ready",
        type=Path,
        help="Marker file created after --export succeeds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print names that would be filled without writing",
    )
    args = parser.parse_args(argv)
    env_path = args.env
    if args.dry_run:
        source = env_path if env_path.exists() else args.example
        _, filled = fill_env_text(source.read_text())
        if filled:
            print("Would fill: " + ", ".join(filled))
        else:
            print("No placeholder service secrets to fill.")
        return 0
    if args.export is None:
        filled = fill_env_file(env_path, example_path=args.example)
        if filled:
            print("Filled: " + ", ".join(filled))
        else:
            print(f"No placeholder service secrets in {env_path}.")
        return 0
    secrets_map, filled, origin = resolve_service_secrets(
        env_path,
        example_path=args.example,
        environ=dict(os.environ),
    )
    write_runtime_secrets_file(secrets_map, args.export)
    if args.ready is not None:
        args.ready.parent.mkdir(parents=True, exist_ok=True)
        args.ready.write_text("ready\n")
    if filled:
        print("Filled: " + ", ".join(filled))
    elif origin == "environ":
        print("Exported service secrets from the process environment.")
    else:
        print(f"No placeholder service secrets in {env_path}.")
    print(f"Exported {len(secrets_map)} service secrets to {args.export}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
