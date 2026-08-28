"""Apply the dedicated Alembic migrations to the runtime database."""

import os
import subprocess


def _run_alembic(*args: str) -> None:
    subprocess.run(["alembic", *args], check=True)


def main() -> None:
    if os.environ.get("RUN_RUNTIME_DB_MIGRATIONS", "true").strip().lower() in {"0", "false", "no", "off"}:
        print("Runtime database migrations are disabled because RUN_RUNTIME_DB_MIGRATIONS=false.", flush=True)
        return

    database_url = os.environ.get("AGENT_RUNTIME_EXECUTION_DATABASE_URL")
    if not database_url:
        raise RuntimeError("AGENT_RUNTIME_EXECUTION_DATABASE_URL is required")

    os.environ["DATABASE_URL"] = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    _run_alembic("-c", "runtime_alembic.ini", "upgrade", "head")


if __name__ == "__main__":
    main()
