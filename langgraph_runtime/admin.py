"""Explicit administrative operations for runtime-owned checkpoint state."""

from __future__ import annotations

import argparse
import asyncio
import json

from langgraph_runtime.checkpointing import delete_agent_checkpoints


def main() -> int:
    parser = argparse.ArgumentParser(description="Administer langgraph-runtime checkpoint state")
    parser.add_argument("--delete-thread", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    targets = sorted({str(value).strip() for value in args.delete_thread if str(value).strip()})
    if targets and not args.dry_run and not args.confirm:
        parser.error("checkpoint deletion requires --confirm or --dry-run")
    deleted = [] if args.dry_run else asyncio.run(delete_agent_checkpoints(targets))
    print(json.dumps({"targets": targets, "deleted": deleted, "dry_run": args.dry_run}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
